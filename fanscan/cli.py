"""Command-line entry point: ``python -m fanscan.cli <subcommand> ...``.

Subcommands
-----------
synth     render a synthetic impeller photo (+ ``<img>.truth.json``)
analyze   photo -> ImpellerParams JSON (+ optional overlay PNG)
build     ImpellerParams JSON (or a synth truth JSON) -> STEP (+ optional STL)
pipeline  synth -> analyze -> build in one directory and score the result

``fanscan.analyze`` is imported lazily inside the subcommands that need it so
that ``synth`` and ``build`` keep working even when the analyzer (or its heavy
dependencies) is unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .params import DEFAULT_PARAMS, ImpellerParams

# Pass/fail thresholds used by ``pipeline`` (see SPEC).
DIAMETER_TOL_PCT = 3.0
BETA_TOL_DEG = 4.0
_DIAMETER_KEYS = ("outer_diameter", "inner_diameter", "hub_diameter")
_BETA_KEYS = ("blade_angle_inlet", "blade_angle_outlet")


# ---- helpers ---------------------------------------------------------------
def load_params(path: str | Path | None) -> ImpellerParams:
    """Load ``ImpellerParams`` from a params JSON *or* a synth truth JSON.

    A truth JSON (written by ``synth``) nests the parameters under the
    ``"params"`` key; a bare params JSON has the fields at the top level.
    ``None`` returns :data:`DEFAULT_PARAMS`.
    """
    if path is None:
        return DEFAULT_PARAMS
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if isinstance(data.get("params"), dict):
        data = data["params"]
    return ImpellerParams.from_dict(data)


def _read_gray(path: str | Path):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def _scale_kwargs(ref_diameter_mm: float | None, outer_diameter_mm: float | None) -> dict[str, float]:
    if (ref_diameter_mm is None) == (outer_diameter_mm is None):
        raise SystemExit("error: give exactly one of --ref-diameter-mm / --outer-diameter-mm")
    if ref_diameter_mm is not None:
        return {"ref_diameter_mm": float(ref_diameter_mm)}
    return {"outer_diameter_mm": float(outer_diameter_mm)}


def _result_summary(result: Any) -> dict[str, Any]:
    """JSON-serialisable summary of an ``AnalysisResult`` (no debug arrays)."""
    return {
        "params": result.params.to_dict(),
        "px_per_mm": float(result.px_per_mm),
        "center_xy": [float(v) for v in result.center_xy],
        "confidence": {k: float(v) for k, v in dict(result.confidence).items()},
    }


def compare_table(truth: ImpellerParams, est: ImpellerParams) -> tuple[str, list[str]]:
    """Render the truth-vs-estimate table; returns (text, list of failure messages)."""
    cmp = truth.compare(est)
    failures: list[str] = []
    rows = [f"{'parameter':<20}{'truth':>10}{'est':>10}{'error':>12}  status"]
    for name, c in cmp.items():
        if name == "blade_count":
            err_txt = f"{c['abs_err']:+d}"
            ok = c["abs_err"] == 0
            if not ok:
                failures.append(f"blade_count mismatch ({c['truth']} vs {c['est']})")
            rows.append(f"{name:<20}{c['truth']:>10d}{c['est']:>10d}{err_txt:>12}  {'ok' if ok else 'FAIL'}")
            continue
        if name in _BETA_KEYS:
            err = c["abs_err_deg"]
            err_txt = f"{err:+.2f} deg"
            ok = abs(err) <= BETA_TOL_DEG
            if not ok:
                failures.append(f"{name} error {err:+.2f} deg exceeds {BETA_TOL_DEG} deg")
            status = "ok" if ok else "FAIL"
        else:
            err = c["rel_err_pct"]
            err_txt = f"{err:+.2f} %"
            if name in _DIAMETER_KEYS:
                ok = abs(err) <= DIAMETER_TOL_PCT
                if not ok:
                    failures.append(f"{name} error {err:+.2f} % exceeds {DIAMETER_TOL_PCT} %")
                status = "ok" if ok else "FAIL"
            else:
                status = "info"  # blade_width / thickness are not pass/fail criteria
        rows.append(f"{name:<20}{c['truth']:>10.2f}{c['est']:>10.2f}{err_txt:>12}  {status}")
    return "\n".join(rows), failures


# ---- subcommands -------------------------------------------------------------
def cmd_synth(a: argparse.Namespace) -> int:
    from . import synth

    params = load_params(a.params)
    img, truth = synth.render(params, px_per_mm=a.px_per_mm, ref_diameter_mm=a.ref_diameter_mm,
                              noise_sigma=a.noise, blur_sigma=a.blur, rotation_deg=a.rotation,
                              seed=a.seed)
    out = synth.save(a.out, img)
    truth_path = synth.save_truth(out.with_suffix(".truth.json"), truth)
    print(f"wrote {out} ({img.shape[1]}x{img.shape[0]}) and {truth_path}")
    return 0


def cmd_analyze(a: argparse.Namespace) -> int:
    from . import analyze as an  # lazy: keeps synth/build usable without it

    img = _read_gray(a.image)
    result = an.analyze(img, blade_width_mm=a.blade_width_mm,
                        blade_count_hint=a.blade_count_hint,
                        **_scale_kwargs(a.ref_diameter_mm, a.outer_diameter_mm))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.params.save(out)
    print(json.dumps(_result_summary(result), indent=2))
    if a.overlay:
        import cv2

        overlay = an.draw_overlay(img, result)
        Path(a.overlay).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(a.overlay), overlay):
            raise OSError(f"could not write overlay to {a.overlay}")
        print(f"wrote overlay {a.overlay}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)
    return 0


def cmd_build(a: argparse.Namespace) -> int:
    from . import cad

    params = load_params(a.params)
    params.validate()
    step = cad.export_step(params, a.out)
    print(f"wrote {step}")
    if a.stl:
        print(f"wrote {cad.export_stl(params, a.stl)}")
    return 0


def cmd_pipeline(a: argparse.Namespace) -> int:
    from . import synth
    from . import analyze as an
    from . import cad
    import cv2

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_params = load_params(a.params)

    # 1. synth
    img, truth = synth.render(truth_params, px_per_mm=a.px_per_mm, ref_diameter_mm=a.ref_diameter_mm,
                              noise_sigma=a.noise, blur_sigma=a.blur, rotation_deg=a.rotation,
                              seed=a.seed)
    img_path = synth.save(out_dir / "synth.png", img)
    synth.save_truth(img_path.with_suffix(".truth.json"), truth)

    # 2. analyze (blade width is not observable in an axial photo: pass it through)
    blade_width = a.blade_width_mm if a.blade_width_mm is not None else truth_params.blade_width
    result = an.analyze(img, ref_diameter_mm=a.ref_diameter_mm, blade_width_mm=blade_width,
                        blade_count_hint=a.blade_count_hint)
    est = result.params
    est.save(out_dir / "params.json")
    cv2.imwrite(str(out_dir / "overlay.png"), an.draw_overlay(img, result))

    # 3. build
    step_path = cad.export_step(est, out_dir / "impeller.step")
    if a.stl:
        cad.export_stl(est, out_dir / "impeller.stl")

    table, failures = compare_table(truth_params, est)
    print(f"pipeline outputs in {out_dir}: synth.png, synth.truth.json, params.json, overlay.png, "
          f"{step_path.name}" + (", impeller.stl" if a.stl else ""))
    print(f"px_per_mm: truth {truth['px_per_mm']:.4f}  est {result.px_per_mm:.4f}")
    print(table)
    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


# ---- argument parsing --------------------------------------------------------
def _add_synth_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--params", help="ImpellerParams JSON or synth truth JSON (default: built-in)")
    p.add_argument("--px-per-mm", type=float, default=4.0)
    p.add_argument("--ref-diameter-mm", type=float, default=30.0,
                   help="physical diameter of the white reference circle")
    p.add_argument("--noise", type=float, default=0.0, help="additive Gaussian noise sigma")
    p.add_argument("--blur", type=float, default=0.0, help="Gaussian blur sigma [px]")
    p.add_argument("--rotation", type=float, default=0.0, help="impeller rotation [deg]")
    p.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m fanscan.cli",
                                 description="fanscan: impeller photo -> parameters -> STEP")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="render a synthetic impeller photo + truth JSON")
    p.add_argument("--out", required=True, help="output PNG (truth JSON written next to it)")
    _add_synth_opts(p)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("analyze", help="estimate impeller parameters from a photo")
    p.add_argument("image", help="input PNG/JPG (axial view)")
    p.add_argument("--out", required=True, help="output params JSON")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref-diameter-mm", type=float, help="diameter of the white reference circle")
    g.add_argument("--outer-diameter-mm", type=float, help="known impeller outer diameter D2")
    p.add_argument("--blade-width-mm", type=float, default=None,
                   help="axial blade width (not observable in the photo)")
    p.add_argument("--blade-count-hint", type=int, default=None)
    p.add_argument("--overlay", help="write a debug overlay PNG")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("build", help="build a STEP model from a params JSON")
    p.add_argument("params", help="ImpellerParams JSON (a synth *.truth.json is accepted too)")
    p.add_argument("--out", required=True, help="output STEP path")
    p.add_argument("--stl", help="optional STL output path")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("pipeline", help="synth -> analyze -> build and score the result")
    p.add_argument("--out-dir", required=True)
    _add_synth_opts(p)
    p.add_argument("--blade-width-mm", type=float, default=None,
                   help="blade width passed to the analyzer (default: the truth value)")
    p.add_argument("--blade-count-hint", type=int, default=None)
    p.add_argument("--stl", action="store_true", help="also write impeller.stl")
    p.set_defaults(func=cmd_pipeline)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
