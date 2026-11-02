"""Command-line entry point: ``python -m fanscan.cli <subcommand> ...``.

Subcommands
-----------
synth       render a v0.1 synthetic axial photo (+ ``<img>.truth.json``)
analyze     photo -> ImpellerParams JSON (+ optional overlay PNG); ``--mode open|shrouded|auto``
build       ImpellerParams JSON (or a truth JSON) -> STEP (+ optional STL)
pipeline    v0.1: synth -> analyze -> build in one directory and score the result
render      v0.2: photo-like 3-D render of an impeller with a camera preset (+ truth JSON)
pipeline3d  v0.2: render -> analyze_shrouded -> build and score the result

Heavy modules (``analyze``, ``shrouded``, ``render3d``, ``cad``) are imported
lazily inside the subcommands that need them, so every other subcommand keeps
working when one of them (or its dependencies) is unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .params import DEFAULT_PARAMS, SHROUDED_PARAMS_FACTORY, ImpellerParams

# Pass/fail thresholds used by ``pipeline`` / ``pipeline3d`` (see SPEC / SPEC2).
DIAMETER_TOL_PCT = 3.0
BETA_TOL_DEG = 4.0
_DIAMETER_KEYS = ("outer_diameter", "inner_diameter", "hub_diameter", "eye_diameter")
_BETA_KEYS = ("blade_angle_inlet", "blade_angle_outlet")

# render3d presets (kept here so the parser does not need to import render3d).
PRESETS_3D = ("axial_far", "axial_close", "tilt", "oblique", "side")
MODES = ("auto", "open", "shrouded")
# Provenance sources that are *not* measurements: reported as info, never pass/fail.
_INFO_SOURCES = ("prior", "user")


# ---- helpers ---------------------------------------------------------------
def load_params(path: str | Path | None, default: ImpellerParams | None = None) -> ImpellerParams:
    """Load ``ImpellerParams`` from a params JSON *or* a truth JSON.

    A truth JSON (written by ``synth`` / ``render``) nests the parameters under
    the ``"params"`` key; a bare params JSON has the fields at the top level.
    ``None`` returns *default* (:data:`DEFAULT_PARAMS` when not given).
    """
    if path is None:
        return default if default is not None else DEFAULT_PARAMS
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


def _write_image(path: str | Path, img) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write image to {path}")
    return path


def _scale_kwargs(ref_diameter_mm: float | None, outer_diameter_mm: float | None) -> dict[str, float]:
    if (ref_diameter_mm is None) == (outer_diameter_mm is None):
        raise SystemExit("error: open mode needs exactly one of --ref-diameter-mm / --outer-diameter-mm")
    if ref_diameter_mm is not None:
        return {"ref_diameter_mm": float(ref_diameter_mm)}
    return {"outer_diameter_mm": float(outer_diameter_mm)}


def _import_shrouded():
    """``fanscan.shrouded`` or a clear error (the module is optional at import time)."""
    try:
        from . import shrouded
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise SystemExit(f"error: shrouded mode needs fanscan.shrouded ({exc})")
    return shrouded


def _result_summary(result: Any) -> dict[str, Any]:
    """JSON-serialisable summary of an ``AnalysisResult`` (no debug arrays)."""
    return {
        "mode": "open",
        "params": result.params.to_dict(),
        "px_per_mm": float(result.px_per_mm),
        "center_xy": [float(v) for v in result.center_xy],
        "confidence": {k: float(v) for k, v in dict(result.confidence).items()},
    }


def _prov_fields(p: Any) -> tuple[float | None, str, float, str]:
    """(value, source, confidence, note) of a Provenance-like object or dict."""
    if isinstance(p, dict):
        get = p.get
    else:
        def get(k, d=None):
            return getattr(p, k, d)
    v = get("value")
    try:
        v = None if v is None else float(v)
    except (TypeError, ValueError):
        pass
    try:
        conf = float(get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return v, str(get("source", "?")), conf, str(get("note", "") or "")


def provenance_dict(result: Any) -> dict[str, dict[str, Any]]:
    """``{field: {value, source, confidence, note}}`` from a ``ShroudedResult``."""
    out: dict[str, dict[str, Any]] = {}
    for name, p in dict(getattr(result, "provenance", {}) or {}).items():
        v, src, conf, note = _prov_fields(p)
        out[str(name)] = {"value": v, "source": src, "confidence": conf, "note": note}
    return out


def provenance_table(result: Any) -> str:
    """Human-readable provenance table (parameter, value, source, confidence, note)."""
    prov = provenance_dict(result)
    rows = [f"{'parameter':<22}{'value':>10}  {'source':<13}{'conf':>5}  note"]
    if not prov:
        rows.append("(no provenance reported)")
        return "\n".join(rows)
    for name, p in prov.items():
        v = p["value"]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v_txt = f"{int(v):d}" if name == "blade_count" and float(v).is_integer() else f"{v:.2f}"
        else:
            v_txt = "-" if v is None else str(v)
        rows.append(f"{name:<22}{v_txt:>10}  {p['source']:<13}{p['confidence']:>5.2f}  {p['note']}".rstrip())
    return "\n".join(rows)


def _shrouded_summary(result: Any) -> dict[str, Any]:
    """JSON-serialisable summary of a ``ShroudedResult``."""
    def ell(e):
        if e is None:
            return None
        to_dict = getattr(e, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return {"center": list(map(float, e.center)), "axes": list(map(float, e.axes)),
                "angle_deg": float(e.angle_deg), "score": float(getattr(e, "score", 0.0))}

    params = result.params
    return {
        "mode": "shrouded",
        "params": params.to_dict(),
        "provenance": provenance_dict(result),
        "rim": ell(getattr(result, "rim", None)),
        "eye": ell(getattr(result, "eye", None)),
        "warnings": list(params.meta.get("warnings", [])) if isinstance(params.meta, dict) else [],
    }


def detect_mode(img, ref_diameter_mm: float | None = None) -> str:
    """``'shrouded'`` when a rim + eye ellipse pair is found and the v0.1 white
    reference disc is absent, else ``'open'`` (the ``--mode auto`` rule)."""
    from . import pose

    if ref_diameter_mm is not None:
        try:
            from . import analyze as an
            finder = getattr(an, "_find_reference_circle", None)
            ref_found = True if finder is None else finder(an._to_gray(img), {}) is not None
        except Exception:  # pragma: no cover - the analyzer is optional here
            ref_found = True
        if ref_found:
            return "open"
    ells = pose.find_ellipses(img)
    rim, eye = pose.pick_impeller_ellipses(ells, img.shape)
    return "shrouded" if (rim is not None and eye is not None) else "open"


def compare_table(truth: ImpellerParams, est: ImpellerParams, *,
                  provenance: dict[str, Any] | None = None,
                  wrap_used: bool = False, mode: str = "open") -> tuple[str, list[str]]:
    """Render the truth-vs-estimate table; returns (text, list of failure messages).

    ``mode='open'`` reproduces the v0.1 rules (blade_count exact, diameters within
    DIAMETER_TOL_PCT, both betas within BETA_TOL_DEG).  ``mode='shrouded'`` adds the
    eye_diameter row, reports beta2 as *info* unless ``wrap_used`` (rim images gave a
    wrap-angle beta2), and reports any field whose provenance source is 'prior' or
    'user' as info (a prior is not a measurement; a user value trivially matches).
    """
    cmp = truth.compare(est)
    if mode == "shrouded" and truth.eye_diameter and est.eye_diameter:
        a, b = float(truth.eye_diameter), float(est.eye_diameter)
        cmp["eye_diameter"] = {"truth": a, "est": b, "abs_err": b - a,
                               "rel_err_pct": 100.0 * (b - a) / a}
    prov = provenance or {}

    def source_of(name: str) -> str:
        p = prov.get(name)
        return "" if p is None else _prov_fields(p)[1]

    failures: list[str] = []
    rows = [f"{'parameter':<20}{'truth':>10}{'est':>10}{'error':>12}  status"]
    for name, c in cmp.items():
        src = source_of(name)
        info_reason = src if (mode == "shrouded" and src in _INFO_SOURCES) else None
        if name == "blade_count":
            err_txt = f"{c['abs_err']:+d}"
            ok = c["abs_err"] == 0
            if info_reason:
                status = f"info ({info_reason})"
            else:
                status = "ok" if ok else "FAIL"
                if not ok:
                    failures.append(f"blade_count mismatch ({c['truth']} vs {c['est']})")
            rows.append(f"{name:<20}{c['truth']:>10d}{c['est']:>10d}{err_txt:>12}  {status}")
            continue
        if name in _BETA_KEYS:
            err = c["abs_err_deg"]
            err_txt = f"{err:+.2f} deg"
            ok = abs(err) <= BETA_TOL_DEG
            if info_reason:
                status = f"info ({info_reason})"
            elif mode == "shrouded" and name == "blade_angle_outlet" and not wrap_used:
                status = "info (extrapolated; give rim images for a wrap-angle beta2)"
            else:
                status = "ok" if ok else "FAIL"
                if not ok:
                    failures.append(f"{name} error {err:+.2f} deg exceeds {BETA_TOL_DEG} deg")
        else:
            err = c["rel_err_pct"]
            err_txt = f"{err:+.2f} %"
            if name in _DIAMETER_KEYS:
                ok = abs(err) <= DIAMETER_TOL_PCT
                if info_reason:
                    status = f"info ({info_reason})"
                else:
                    status = "ok" if ok else "FAIL"
                    if not ok:
                        failures.append(f"{name} error {err:+.2f} % exceeds {DIAMETER_TOL_PCT} %")
            else:
                status = "info"  # blade_width / thickness are not pass/fail criteria
                if info_reason:
                    status = f"info ({info_reason})"
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


def _analyze_open(a: argparse.Namespace, img) -> tuple[Any, dict[str, Any], Any]:
    from . import analyze as an

    ignored = [opt for opt, val in (("--eye-diameter-mm", a.eye_diameter_mm),
                                    ("--blade-width-inlet-mm", a.blade_width_inlet_mm),
                                    ("--rim-image", a.rim_image), ("--marker", a.marker)) if val]
    if ignored:
        print(f"warning: open mode ignores {', '.join(ignored)}", file=sys.stderr)
    result = an.analyze(img, blade_width_mm=a.blade_width_mm,
                        blade_count_hint=a.blade_count_hint,
                        **_scale_kwargs(a.ref_diameter_mm, a.outer_diameter_mm))
    return result, _result_summary(result), an.draw_overlay


def _analyze_shrouded(a: argparse.Namespace, img) -> tuple[Any, dict[str, Any], Any]:
    sh = _import_shrouded()
    rim_images = [_read_gray(p) for p in (a.rim_image or [])]
    result = sh.analyze_shrouded(
        img,
        outer_diameter_mm=a.outer_diameter_mm, eye_diameter_mm=a.eye_diameter_mm,
        ref_diameter_mm=a.ref_diameter_mm, blade_width_mm=a.blade_width_mm,
        blade_width_inlet_mm=a.blade_width_inlet_mm, blade_count_hint=a.blade_count_hint,
        rim_images=rim_images or None, marker_hint=a.marker,
    )
    return result, _shrouded_summary(result), sh.draw_overlay


def cmd_analyze(a: argparse.Namespace) -> int:
    img = _read_gray(a.image)
    mode = a.mode
    if mode == "auto":
        mode = detect_mode(img, a.ref_diameter_mm)
        print(f"auto mode -> {mode}", file=sys.stderr)
    if mode == "open":
        result, summary, draw_overlay = _analyze_open(a, img)
    else:
        result, summary, draw_overlay = _analyze_shrouded(a, img)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.params.save(out)
    print(json.dumps(summary, indent=2))
    if mode == "shrouded":
        print(provenance_table(result))
    if a.overlay:
        _write_image(a.overlay, draw_overlay(img, result))
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


# ---- v0.2: render / pipeline3d ------------------------------------------------
def _render_kwargs(a: argparse.Namespace, **extra) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "size": (a.size, a.size), "clutter": a.clutter, "noise_sigma": a.noise,
        "blur_sigma": a.blur, "seed": a.seed, "tilt_deg": a.tilt_deg,
        "marker": bool(getattr(a, "marker", False)), "marker_deg": a.marker_deg,
    }
    if getattr(a, "azimuth_deg", None) is not None:
        kw["azimuth_deg"] = a.azimuth_deg
    if getattr(a, "elevation_deg", None) is not None:
        kw["elevation_deg"] = a.elevation_deg
    if getattr(a, "distance", None) is not None:
        kw["distance"] = a.distance
    kw.update(extra)
    return kw


def _check_preset(name: str) -> str:
    if name not in PRESETS_3D:
        raise SystemExit(f"error: unknown preset {name!r}; choose from {', '.join(PRESETS_3D)}")
    return name


def cmd_render(a: argparse.Namespace) -> int:
    from . import render3d

    params = load_params(a.params, default=SHROUDED_PARAMS_FACTORY())
    params.validate()
    preset = _check_preset(a.preset)
    img, truth = render3d.render_params(params, preset, **_render_kwargs(a))
    out = render3d.save(a.out, img)
    truth_path = out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(render3d.truth_json(truth), indent=1))
    tm = truth.get("timing_s", {})
    rim = truth.get("rim_ellipse") or {}
    print(f"wrote {out} ({img.shape[1]}x{img.shape[0]}) and {truth_path}")
    print(f"preset {preset}: {truth.get('n_triangles', '?')} triangles, "
          f"render {tm.get('total', float('nan')):.2f} s, px_per_mm {truth.get('px_per_mm', float('nan')):.3f}, "
          f"rim a/b {rim.get('axes', ['?', '?'])[0]}/{rim.get('axes', ['?', '?'])[1]} px, "
          f"visible blade fraction {truth.get('visible_blade_fraction', float('nan')):.2f}")
    return 0


def _parse_azimuths(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(t) for t in text.replace(";", ",").split(",") if t.strip()]


def cmd_pipeline3d(a: argparse.Namespace) -> int:
    from . import render3d

    sh = _import_shrouded()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_params = load_params(a.params, default=SHROUDED_PARAMS_FACTORY())
    truth_params.validate()
    preset = _check_preset(a.preset)
    azimuths = _parse_azimuths(a.rim_azimuths)
    use_marker = bool(a.marker or azimuths)

    # 1. render the main (axial-ish) view, plus optional side views of the rim gap
    img, truth = render3d.render_params(truth_params, preset, **_render_kwargs(a, marker=use_marker))
    img_path = render3d.save(out_dir / "render.png", img)
    img_path.with_suffix(".truth.json").write_text(json.dumps(render3d.truth_json(truth), indent=1))
    rim_images = []
    for az in azimuths:
        rim_img, rim_truth = render3d.render_params(
            truth_params, "side", **_render_kwargs(a, marker=True, azimuth_deg=az, tilt_deg=a.tilt_deg))
        p = render3d.save(out_dir / f"rim_az{az:g}.png", rim_img)
        p.with_suffix(".truth.json").write_text(json.dumps(render3d.truth_json(rim_truth), indent=1))
        rim_images.append(rim_img)

    # 2. analyze (what the analyzer is told mirrors what a user could know)
    kw: dict[str, Any] = {"blade_count_hint": a.blade_count_hint}
    if not a.no_outer_diameter:
        kw["outer_diameter_mm"] = float(truth_params.outer_diameter)
    if a.give_eye_diameter:
        kw["eye_diameter_mm"] = float(truth_params.r_eye * 2.0)
    if not a.no_blade_width:
        kw["blade_width_mm"] = float(truth_params.b2)
        kw["blade_width_inlet_mm"] = float(truth_params.b1)
    if rim_images:
        kw["rim_images"] = rim_images
        kw["marker_hint"] = "bright"
    result = sh.analyze_shrouded(img, **kw)
    est = result.params
    est.save(out_dir / "params.json")
    _write_image(out_dir / "overlay.png", sh.draw_overlay(img, result))
    (out_dir / "provenance.json").write_text(json.dumps(provenance_dict(result), indent=2))

    # 3. build
    built = []
    if not a.skip_build:
        from . import cad

        est.validate()
        built.append(cad.export_step(est, out_dir / "impeller.step").name)
        if a.stl:
            built.append(cad.export_stl(est, out_dir / "impeller.stl").name)

    prov = provenance_dict(result)
    wrap_used = prov.get("blade_angle_outlet", {}).get("source") == "wrap"
    table, failures = compare_table(truth_params, est, provenance=prov, wrap_used=wrap_used,
                                    mode="shrouded")
    files = ["render.png", "render.truth.json"] + [f"rim_az{az:g}.png" for az in azimuths] + \
        ["params.json", "overlay.png", "provenance.json"] + built
    print(f"pipeline3d outputs in {out_dir}: {', '.join(files)}")
    print(f"preset {preset}{' + clutter' if a.clutter else ''}, truth visible blade fraction "
          f"{truth.get('visible_blade_fraction', float('nan')):.2f}, px_per_mm (backplate plane) "
          f"{truth.get('px_per_mm', float('nan')):.3f}"
          + (f", rim images at azimuths {azimuths}" if azimuths else ""))
    print(provenance_table(result))
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


def _add_render_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="axial_close", help=f"camera preset: {', '.join(PRESETS_3D)}")
    p.add_argument("--params", help="ImpellerParams JSON or truth JSON (default: SHROUDED_PARAMS_FACTORY)")
    p.add_argument("--clutter", action="store_true", help="procedural desk-clutter background")
    p.add_argument("--noise", type=float, default=0.0, help="additive Gaussian noise sigma")
    p.add_argument("--blur", type=float, default=0.8, help="Gaussian blur sigma [px]")
    p.add_argument("--tilt-deg", type=float, default=15.0, help="axis tilt for the 'tilt' preset")
    p.add_argument("--size", type=int, default=1200, help="square image size [px]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--marker", action="store_true", help="add a bright marker cube on the shroud rim")
    p.add_argument("--marker-deg", type=float, default=270.0, help="marker azimuth [deg]")
    p.add_argument("--azimuth-deg", type=float, default=None, help="camera azimuth override [deg]")
    p.add_argument("--elevation-deg", type=float, default=None, help="camera elevation override [deg]")
    p.add_argument("--distance", type=float, default=None, help="camera distance override [mm]")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m fanscan.cli",
                                 description="fanscan: impeller photo -> parameters -> STEP")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="render a synthetic impeller photo + truth JSON")
    p.add_argument("--out", required=True, help="output PNG (truth JSON written next to it)")
    _add_synth_opts(p)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("analyze", help="estimate impeller parameters from a photo")
    p.add_argument("image", help="input PNG/JPG (axial / through-the-eye view)")
    p.add_argument("--out", required=True, help="output params JSON")
    p.add_argument("--mode", choices=MODES, default="auto",
                   help="open = v0.1 analyzer; shrouded = v0.2 closed-impeller analyzer; "
                        "auto = shrouded when a rim+eye ellipse pair is found and no reference disc")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--ref-diameter-mm", type=float, help="diameter of the white reference circle")
    g.add_argument("--outer-diameter-mm", type=float, help="known impeller outer diameter D2")
    p.add_argument("--eye-diameter-mm", type=float, default=None,
                   help="(shrouded) known inlet-ring inner diameter; scale fallback / perspective")
    p.add_argument("--blade-width-mm", type=float, default=None,
                   help="axial blade width b (b2 at the rim; not observable in an axial photo)")
    p.add_argument("--blade-width-inlet-mm", type=float, default=None,
                   help="(shrouded) blade width b1 at the leading edge")
    p.add_argument("--blade-count-hint", type=int, default=None)
    p.add_argument("--rim-image", action="append", default=[], metavar="IMG",
                   help="(shrouded) side view of the rim gap with the marker visible; repeatable")
    p.add_argument("--marker", choices=("bright", "dark"), default=None,
                   help="(shrouded) kind of azimuth marker on the shroud rim")
    p.add_argument("--overlay", help="write a debug overlay PNG")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("build", help="build a STEP model from a params JSON")
    p.add_argument("params", help="ImpellerParams JSON (a *.truth.json is accepted too)")
    p.add_argument("--out", required=True, help="output STEP path")
    p.add_argument("--stl", help="optional STL output path")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("pipeline", help="synth -> analyze -> build and score the result (v0.1)")
    p.add_argument("--out-dir", required=True)
    _add_synth_opts(p)
    p.add_argument("--blade-width-mm", type=float, default=None,
                   help="blade width passed to the analyzer (default: the truth value)")
    p.add_argument("--blade-count-hint", type=int, default=None)
    p.add_argument("--stl", action="store_true", help="also write impeller.stl")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("render", help="photo-like 3-D render with a camera preset + truth JSON (v0.2)")
    p.add_argument("--out", required=True, help="output PNG (truth JSON written next to it)")
    _add_render_opts(p)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("pipeline3d", help="render -> analyze_shrouded -> build and score (v0.2)")
    p.add_argument("--out-dir", required=True)
    _add_render_opts(p)
    p.add_argument("--rim-azimuths", default=None, metavar="A,B,...",
                   help="also render 'side' views at these camera azimuths with a marker and pass them "
                        "as rim images (wrap-angle beta2 becomes pass/fail), e.g. 240,270,300")
    p.add_argument("--no-outer-diameter", action="store_true",
                   help="do not tell the analyzer D2 (exercise the prior / unknown-scale path)")
    p.add_argument("--give-eye-diameter", action="store_true",
                   help="also tell the analyzer D_eye (enables the perspective correction)")
    p.add_argument("--no-blade-width", action="store_true",
                   help="do not tell the analyzer b1/b2 (exercise the prior path)")
    p.add_argument("--blade-count-hint", type=int, default=None)
    p.add_argument("--skip-build", action="store_true", help="skip the STEP build")
    p.add_argument("--stl", action="store_true", help="also write impeller.stl")
    p.set_defaults(func=cmd_pipeline3d)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
