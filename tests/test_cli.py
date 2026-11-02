"""CLI contract tests for ``fanscan.cli`` (fast: no 3-D render, no CAD, analyzers mocked).

Contract (see the module docstring of fanscan/cli.py):
* length / scale options must be > 0, noise / blur / seed >= 0 -> argparse usage error, exit 2;
* user-input and analysis failures -> one ``error: ...`` line on stderr, exit 1, no traceback;
* ``load_params`` reads a bare params JSON or a truth JSON (``"params"`` nested);
* ``analyze`` / ``pipeline`` / ``pipeline3d`` validate the analyzer output before writing it;
* ``pipeline`` passes the truth D2 as a fallback scale next to the reference-disc diameter;
* the shrouded path forwards focal_px / hub_height_mm / handedness to ``analyze_shrouded``
  and prints the provenance table;
* ``pipeline3d`` tells the analyzer D2 and D_eye by default and renders the 'realistic'
  variant (D1 = 120 mm, hub flange 20 mm) unless ``--params`` / ``--variant`` say otherwise.
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fanscan import cli
from fanscan.params import DEFAULT_PARAMS, SHROUDED_PARAMS_FACTORY, ImpellerParams


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_png(tmp_path):
    """A small readable grayscale image (the mocked analyzers never look at it)."""
    img = np.full((64, 80), 128, np.uint8)
    p = tmp_path / "tiny.png"
    cv2.imwrite(str(p), img)
    return p


def _open_result(params: ImpellerParams):
    from fanscan import analyze as an

    return an.AnalysisResult(params=params, px_per_mm=4.0, center_xy=(10.0, 12.0),
                             confidence={"blade_count": 1.0})


def _shrouded_result(params: ImpellerParams):
    from fanscan import shrouded

    prov = {
        "outer_diameter": shrouded.Provenance(params.outer_diameter, "user", 1.0, "given"),
        "eye_diameter": shrouded.Provenance(params.r_eye * 2.0, "measured", 0.5, "ring"),
        "hub_diameter": shrouded.Provenance(params.hub_diameter, "measured", 0.4, ""),
        "blade_count": shrouded.Provenance(params.blade_count, "measured", 0.9, "fft"),
        "inner_diameter": shrouded.Provenance(params.inner_diameter, "prior", 0.2, ""),
        "blade_angle_inlet": shrouded.Provenance(params.blade_angle_inlet, "measured", 0.6, ""),
        "blade_angle_outlet": shrouded.Provenance(params.blade_angle_outlet, "extrapolated", 0.3, ""),
        "blade_width": shrouded.Provenance(params.blade_width, "user", 1.0, "given"),
    }
    return SimpleNamespace(params=params, provenance=prov, rim=None, eye=None, rectified=None, debug={})


def _err(capsys) -> str:
    return capsys.readouterr().err


# ---------------------------------------------------------------------------
# argparse value types: bad values never reach the analyzers (usage error, exit 2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("argv", [
    ["analyze", "x.png", "--out", "p.json", "--ref-diameter-mm", "0"],
    ["analyze", "x.png", "--out", "p.json", "--ref-diameter-mm", "-30"],
    ["analyze", "x.png", "--out", "p.json", "--outer-diameter-mm", "0"],
    ["analyze", "x.png", "--out", "p.json", "--outer-diameter-mm", "nan"],
    ["analyze", "x.png", "--out", "p.json", "--ref-diameter-mm", "30", "--blade-width-mm", "-5"],
    ["analyze", "x.png", "--out", "p.json", "--ref-diameter-mm", "30", "--blade-width-mm", "0"],
    ["analyze", "x.png", "--out", "p.json", "--eye-diameter-mm", "-1"],
    ["analyze", "x.png", "--out", "p.json", "--blade-width-inlet-mm", "0"],
    ["analyze", "x.png", "--out", "p.json", "--focal-px", "0"],
    ["analyze", "x.png", "--out", "p.json", "--focal-35mm", "-26"],
    ["analyze", "x.png", "--out", "p.json", "--hub-height-mm", "0"],
    ["analyze", "x.png", "--out", "p.json", "--blade-count-hint", "0"],
    ["analyze", "x.png", "--out", "p.json", "--ref-diameter-mm", "abc"],
    ["analyze", "x.png", "--out", "p.json", "--ref-diameter-mm", "30", "--outer-diameter-mm", "200"],
    ["analyze", "x.png", "--out", "p.json", "--handedness", "left"],
    ["synth", "--out", "x.png", "--seed", "-1"],
    ["synth", "--out", "x.png", "--noise", "-1"],
    ["synth", "--out", "x.png", "--blur", "-0.5"],
    ["synth", "--out", "x.png", "--px-per-mm", "0"],
    ["synth", "--out", "x.png", "--ref-diameter-mm", "-30"],
    ["pipeline", "--out-dir", "d", "--blade-width-mm", "-5"],
    ["pipeline", "--out-dir", "d", "--px-per-mm", "-4"],
    ["render", "--out", "x.png", "--size", "0"],
    ["render", "--out", "x.png", "--distance", "0"],
    ["render", "--out", "x.png", "--preset", "nope"],
    ["render", "--out", "x.png", "--variant", "nope"],
    ["pipeline3d", "--out-dir", "d", "--eye-diameter-mm", "0"],
    ["pipeline3d", "--out-dir", "d", "--outer-diameter-mm", "-300"],
    ["pipeline3d", "--out-dir", "d", "--seed", "-3"],
])
def test_bad_option_values_are_usage_errors(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    assert "error" in _err(capsys)


@pytest.mark.parametrize("fn, good, bad", [
    (cli._positive_float, ["1", "0.5", "1e3"], ["0", "-1", "nan", "inf", "x"]),
    (cli._nonneg_float, ["0", "2.5"], ["-0.1", "-inf", ""]),
    (cli._positive_int, ["1", "11"], ["0", "-2", "1.5", "x"]),
    (cli._nonneg_int, ["0", "3"], ["-1", "2.0"]),
])
def test_value_type_helpers(fn, good, bad):
    import argparse

    for g in good:
        assert fn(g) == float(g)
    for b in bad:
        with pytest.raises(argparse.ArgumentTypeError):
            fn(b)


def test_scale_kwargs_rules():
    assert cli._scale_kwargs(30.0, None) == {"ref_diameter_mm": 30.0}
    assert cli._scale_kwargs(None, 200) == {"outer_diameter_mm": 200.0}
    with pytest.raises(cli.CliError):
        cli._scale_kwargs(None, None)
    with pytest.raises(cli.CliError):
        cli._scale_kwargs(30.0, 200.0)
    with pytest.raises(cli.CliError):
        cli._scale_kwargs(0.0, None)
    with pytest.raises(cli.CliError):
        cli._scale_kwargs(None, -5.0)


# ---------------------------------------------------------------------------
# load_params: bare params JSON, truth JSON, and the failure messages
# ---------------------------------------------------------------------------
def test_load_params_bare_and_truth_json(tmp_path):
    p = SHROUDED_PARAMS_FACTORY()
    bare = tmp_path / "bare.json"
    p.save(bare)                                   # includes the "_derived" block -> ignored
    truth = tmp_path / "shot.truth.json"
    truth.write_text(json.dumps({"preset": "axial_close", "px_per_mm": 3.0, "params": p.to_dict()}))
    for path in (bare, truth, str(bare)):
        q = cli.load_params(path)
        assert dataclasses.asdict(q) == dataclasses.asdict(p)
    assert cli.load_params(None) is DEFAULT_PARAMS
    d = cli.realistic_params()
    assert cli.load_params(None, default=d) is d


@pytest.mark.parametrize("content, needle", [
    ("{oops", "not valid JSON"),
    ("[1, 2]", "expected a JSON object"),
    ('{"outer_diameter": 200}', "missing fields"),
    ('{"params": {"outer_diameter": 200, "blade_count": 9}}', "inner_diameter"),
])
def test_load_params_errors(tmp_path, content, needle):
    f = tmp_path / "bad.json"
    f.write_text(content)
    with pytest.raises(cli.CliError) as e:
        cli.load_params(f)
    assert needle in str(e.value) and "bad.json" in str(e.value)


def test_load_params_missing_file(tmp_path):
    with pytest.raises(cli.CliError) as e:
        cli.load_params(tmp_path / "nope.json")
    assert "not found" in str(e.value)


# ---------------------------------------------------------------------------
# main(): user-input errors are one 'error:' line and exit 1, never a traceback
# ---------------------------------------------------------------------------
def test_main_errors_are_clean(tmp_path, capsys):
    out = tmp_path / "x.step"
    # missing params file
    assert cli.main(["build", str(tmp_path / "nope.json"), "--out", str(out)]) == 1
    err = _err(capsys)
    assert err.startswith("error:") and "not found" in err and "Traceback" not in err
    # malformed JSON
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert cli.main(["build", str(bad), "--out", str(out)]) == 1
    assert "error:" in _err(capsys)
    # incomplete params
    bad.write_text(json.dumps({"outer_diameter": 200}))
    assert cli.main(["synth", "--out", str(tmp_path / "s.png"), "--params", str(bad)]) == 1
    err = _err(capsys)
    assert "missing fields" in err and "Traceback" not in err
    # invalid geometry (hub > D1): rejected by validate(), reported cleanly, nothing built
    p = dataclasses.replace(DEFAULT_PARAMS, hub_diameter=150.0)
    bad.write_text(json.dumps(p.to_dict()))
    assert cli.main(["build", str(bad), "--out", str(out)]) == 1
    err = _err(capsys)
    assert "error:" in err and "hub" in err and not out.exists()
    assert cli.main(["synth", "--out", str(tmp_path / "s.png"), "--params", str(bad)]) == 1
    assert "hub" in _err(capsys)
    # missing / unreadable image
    assert cli.main(["analyze", str(tmp_path / "nope.png"), "--out", str(tmp_path / "p.json"),
                     "--ref-diameter-mm", "30"]) == 1
    assert "image not found" in _err(capsys)
    bogus = tmp_path / "bogus.png"
    bogus.write_text("not an image")
    assert cli.main(["analyze", str(bogus), "--out", str(tmp_path / "p.json"),
                     "--ref-diameter-mm", "30"]) == 1
    assert "could not read image" in _err(capsys)


def test_open_mode_needs_exactly_one_scale(tiny_png, tmp_path, capsys):
    rc = cli.main(["analyze", str(tiny_png), "--mode", "open", "--out", str(tmp_path / "p.json")])
    assert rc == 1
    err = _err(capsys)
    assert "error: open mode needs exactly one of" in err and "Traceback" not in err
    assert not (tmp_path / "p.json").exists()


def test_analysis_failure_is_clean(tiny_png, tmp_path, capsys, monkeypatch):
    from fanscan import analyze as an

    def boom(img, **kw):
        raise ValueError("reference circle not found and no outer_diameter_mm given")

    monkeypatch.setattr(an, "analyze", boom)
    rc = cli.main(["analyze", str(tiny_png), "--mode", "open", "--out", str(tmp_path / "p.json"),
                   "--ref-diameter-mm", "30"])
    assert rc == 1
    err = _err(capsys)
    assert err.strip() == "error: reference circle not found and no outer_diameter_mm given"
    assert not (tmp_path / "p.json").exists()


# ---------------------------------------------------------------------------
# analyze: validate before saving; open / shrouded wiring (mocked analyzers)
# ---------------------------------------------------------------------------
def test_analyze_open_writes_validated_params(tiny_png, tmp_path, capsys, monkeypatch):
    from fanscan import analyze as an

    calls = []

    def fake(img, **kw):
        calls.append(kw)
        p = dataclasses.replace(DEFAULT_PARAMS, meta={"warnings": ["blade_width assumed"]})
        return _open_result(p)

    monkeypatch.setattr(an, "analyze", fake)
    monkeypatch.setattr(an, "draw_overlay", lambda img, res: img)
    out, ov = tmp_path / "an" / "p.json", tmp_path / "an" / "o.png"
    rc = cli.main(["analyze", str(tiny_png), "--mode", "open", "--out", str(out), "--overlay", str(ov),
                   "--ref-diameter-mm", "30", "--blade-width-mm", "40", "--blade-count-hint", "9",
                   "--eye-diameter-mm", "180", "--handedness", "cw"])
    assert rc == 0
    assert calls == [{"blade_width_mm": 40.0, "blade_count_hint": 9, "ref_diameter_mm": 30.0}]
    assert out.exists() and ov.exists()
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["mode"] == "open" and summary["warnings"] == ["blade_width assumed"]
    assert summary["params"]["blade_count"] == 9
    assert "open mode ignores --eye-diameter-mm, --handedness" in captured.err
    assert "warning: blade_width assumed" in captured.err
    ImpellerParams.load(out).validate()


def test_analyze_rejects_invalid_analyzer_output(tiny_png, tmp_path, capsys, monkeypatch):
    from fanscan import analyze as an

    monkeypatch.setattr(an, "analyze",
                        lambda img, **kw: _open_result(dataclasses.replace(DEFAULT_PARAMS, blade_width=-5.0)))
    out = tmp_path / "p.json"
    rc = cli.main(["analyze", str(tiny_png), "--mode", "open", "--out", str(out),
                   "--ref-diameter-mm", "30", "--overlay", str(tmp_path / "o.png")])
    assert rc == 1
    err = _err(capsys)
    assert "error: analyzer produced an invalid parameter set" in err and "blade_width" in err
    assert not out.exists() and not (tmp_path / "o.png").exists()


def test_analyze_shrouded_wiring(tiny_png, tmp_path, capsys, monkeypatch):
    from fanscan import shrouded

    calls = []
    p = cli.realistic_params()

    def fake(img, **kw):
        calls.append(kw)
        return _shrouded_result(dataclasses.replace(p, meta={"warnings": ["w1"], "handedness": "cw",
                                                               "visible_blade_fraction": 0.4}))

    monkeypatch.setattr(shrouded, "analyze_shrouded", fake)
    monkeypatch.setattr(shrouded, "draw_overlay", lambda img, res: img)
    out = tmp_path / "s.json"
    rc = cli.main(["analyze", str(tiny_png), "--mode", "shrouded", "--out", str(out),
                   "--outer-diameter-mm", "300", "--eye-diameter-mm", "180", "--blade-width-mm", "55",
                   "--blade-width-inlet-mm", "75", "--blade-count-hint", "11", "--marker", "bright",
                   "--focal-35mm", "27", "--hub-height-mm", "20", "--handedness", "cw"])
    assert rc == 0
    (kw,) = calls
    assert kw["outer_diameter_mm"] == 300.0 and kw["eye_diameter_mm"] == 180.0
    assert kw["ref_diameter_mm"] is None and kw["rim_images"] is None and kw["marker_hint"] == "bright"
    assert kw["blade_width_mm"] == 55.0 and kw["blade_width_inlet_mm"] == 75.0 and kw["blade_count_hint"] == 11
    assert kw["focal_px"] == pytest.approx(27.0 / 36.0 * 80)       # f35 / 36 mm * image width
    assert kw["hub_height_mm"] == 20.0 and kw["handedness"] == "cw"
    captured = capsys.readouterr()
    head, table = captured.out.split("\nparameter", 1)
    summary = json.loads(head)
    assert summary["mode"] == "shrouded" and summary["handedness"] == "cw"
    assert summary["visible_blade_fraction"] == 0.4 and summary["warnings"] == ["w1"]
    assert summary["provenance"]["outer_diameter"] == {"value": 300.0, "source": "user",
                                                        "confidence": 1.0, "note": "given"}
    assert summary["rim"] is None
    assert table.startswith("  ") and "blade_count" in table and "extrapolated" in table
    assert "warning: w1" in captured.err
    assert out.exists()
    # defaults: focal / hub height / handedness are None, rim images read from disk
    rim = tmp_path / "rim.png"
    cv2.imwrite(str(rim), np.zeros((20, 30), np.uint8))
    rc = cli.main(["analyze", str(tiny_png), "--mode", "shrouded", "--out", str(out), "--focal-px", "900",
                   "--rim-image", str(rim), "--rim-image", str(rim)])
    assert rc == 0
    kw = calls[-1]
    assert kw["focal_px"] == 900.0 and kw["hub_height_mm"] is None and kw["handedness"] is None
    assert kw["outer_diameter_mm"] is None and len(kw["rim_images"]) == 2
    assert kw["rim_images"][0].shape == (20, 30)
    capsys.readouterr()
    # a missing rim image is a clean error
    rc = cli.main(["analyze", str(tiny_png), "--mode", "shrouded", "--out", str(out),
                   "--rim-image", str(tmp_path / "nope.png")])
    assert rc == 1 and "image not found" in _err(capsys)


def test_provenance_table_and_dict():
    from fanscan import shrouded

    res = _shrouded_result(cli.realistic_params())
    d = cli.provenance_dict(res)
    assert d["blade_count"] == {"value": 11.0, "source": "measured", "confidence": 0.9, "note": "fft"}
    text = cli.provenance_table(res)
    lines = text.splitlines()
    assert lines[0].startswith("parameter") and "source" in lines[0]
    assert any(line.startswith("blade_count") and " 11 " in line + " " for line in lines)
    assert any(line.startswith("blade_angle_outlet") and "extrapolated" in line for line in lines)
    # the same rows as the library-side formatter
    assert len(lines) == len(res.provenance) + 1
    assert cli.provenance_table(SimpleNamespace(provenance={})).endswith("(no provenance reported)")
    # dict provenance (e.g. provenance.json) is accepted by compare_table
    table, failures = cli.compare_table(cli.realistic_params(), cli.realistic_params(), provenance=d,
                                        mode="shrouded")
    assert not failures and "info (user)" in table and "info (prior)" in table
    assert isinstance(shrouded.Provenance(1, "prior", 0.1), shrouded.Provenance)


# ---------------------------------------------------------------------------
# pipeline (v0.1) with a monkeypatched analyzer and CAD export
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_cad(monkeypatch):
    from fanscan import cad

    def export_step(params, path):
        params.validate()
        from pathlib import Path
        Path(path).write_text("ISO-10303-21;")
        return Path(path)

    monkeypatch.setattr(cad, "export_step", export_step)
    monkeypatch.setattr(cad, "export_stl", export_step)
    return export_step


def _run_pipeline(tmp_path, monkeypatch, make_result, extra=()):
    from fanscan import analyze as an

    calls = []

    def fake(img, **kw):
        calls.append(kw)
        return make_result(kw)

    monkeypatch.setattr(an, "analyze", fake)
    monkeypatch.setattr(an, "draw_overlay", lambda img, res: img)
    out_dir = tmp_path / "pp"
    rc = cli.main(["pipeline", "--out-dir", str(out_dir), "--px-per-mm", "2", *extra])
    return rc, out_dir, calls


def test_pipeline_pass_and_fallback_scale(tmp_path, capsys, monkeypatch, fake_cad):
    rc, out_dir, calls = _run_pipeline(tmp_path, monkeypatch, lambda kw: _open_result(DEFAULT_PARAMS),
                                       extra=["--noise", "30", "--stl"])
    assert rc == 0
    (kw,) = calls
    # both scale sources go to the analyzer: the disc diameter and the truth D2 as fallback
    assert kw["ref_diameter_mm"] == 30.0 and kw["outer_diameter_mm"] == DEFAULT_PARAMS.outer_diameter
    assert kw["blade_width_mm"] == DEFAULT_PARAMS.blade_width and kw["blade_count_hint"] is None
    for name in ("synth.png", "synth.truth.json", "params.json", "overlay.png", "impeller.step",
                 "impeller.stl"):
        assert (out_dir / name).exists(), name
    out = capsys.readouterr().out
    assert out.rstrip().endswith("PASS") and "blade_count" in out


def test_pipeline_fail_exit_code(tmp_path, capsys, monkeypatch, fake_cad):
    off = dataclasses.replace(DEFAULT_PARAMS, outer_diameter=DEFAULT_PARAMS.outer_diameter * 1.1)
    rc, out_dir, _ = _run_pipeline(tmp_path, monkeypatch, lambda kw: _open_result(off),
                                   extra=["--blade-width-mm", "33", "--blade-count-hint", "9"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL:" in out and "outer_diameter error +10.00 % exceeds" in out
    assert (out_dir / "impeller.step").exists()


def test_pipeline_analyzer_error_is_clean(tmp_path, capsys, monkeypatch, fake_cad):
    def boom(kw):
        raise ValueError("reference circle not found and no outer_diameter_mm given")

    rc, out_dir, _ = _run_pipeline(tmp_path, monkeypatch, boom)
    assert rc == 1
    err = _err(capsys)
    assert err.startswith("error: analyze failed on") and "synth.png" in err and "Traceback" not in err
    assert (out_dir / "synth.png").exists() and (out_dir / "synth.truth.json").exists()
    assert not (out_dir / "params.json").exists() and not (out_dir / "impeller.step").exists()


def test_pipeline_invalid_analyzer_output(tmp_path, capsys, monkeypatch, fake_cad):
    bad = dataclasses.replace(DEFAULT_PARAMS, hub_diameter=500.0)
    rc, out_dir, _ = _run_pipeline(tmp_path, monkeypatch, lambda kw: _open_result(bad))
    assert rc == 1
    assert "error: analyzer produced an invalid parameter set" in _err(capsys)
    assert not (out_dir / "params.json").exists()


def test_pipeline_warnings_surface(tmp_path, capsys, monkeypatch, fake_cad):
    p = dataclasses.replace(DEFAULT_PARAMS,
                            meta={"warnings": ["reference circle not found; scale taken from outer_diameter_mm"]})
    rc, _, _ = _run_pipeline(tmp_path, monkeypatch, lambda kw: _open_result(p))
    assert rc == 0
    assert "warning: reference circle not found; scale taken from outer_diameter_mm" in _err(capsys)


# ---------------------------------------------------------------------------
# pipeline3d with a monkeypatched renderer / analyzer
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_render3d(monkeypatch):
    from fanscan import render3d

    calls = []

    def render_params(params, preset, **kw):
        calls.append((dataclasses.replace(params), preset, kw))
        img = np.full((kw["size"][1], kw["size"][0]), 90, np.uint8)
        truth = {"preset": preset, "px_per_mm": 3.0, "visible_blade_fraction": 0.5,
                 "camera": {"focal_px": 1080.0}, "params": params.to_dict(), "_arr": np.zeros(2)}
        return img, truth

    monkeypatch.setattr(render3d, "render_params", render_params)
    return calls


_P3D_RUNS = [0]


def _run_p3d(tmp_path, monkeypatch, make_result, extra=()):
    from fanscan import shrouded

    calls = []

    def fake(img, **kw):
        calls.append(kw)
        return make_result(kw)

    monkeypatch.setattr(shrouded, "analyze_shrouded", fake)
    monkeypatch.setattr(shrouded, "draw_overlay", lambda img, res: img)
    _P3D_RUNS[0] += 1
    out_dir = tmp_path / f"p3d{_P3D_RUNS[0]}"        # a fresh directory per run
    rc = cli.main(["pipeline3d", "--out-dir", str(out_dir), "--size", "64", "--skip-build", *extra])
    return rc, out_dir, calls


def test_pipeline3d_defaults_realistic_with_d2_and_deye(tmp_path, capsys, monkeypatch, fake_render3d):
    rc, out_dir, calls = _run_p3d(tmp_path, monkeypatch, lambda kw: _shrouded_result(cli.realistic_params()))
    assert rc == 0
    truth_p, preset, rkw = fake_render3d[0]
    assert preset == "axial_close" and truth_p.inner_diameter == 120.0 and truth_p.hub_height == 20.0
    assert rkw["marker"] is False and rkw["size"] == (64, 64)
    (kw,) = calls
    assert kw["outer_diameter_mm"] == 300.0 and kw["eye_diameter_mm"] == 180.0
    assert kw["blade_width_mm"] == 55.0 and kw["blade_width_inlet_mm"] == 75.0
    assert "focal_px" not in kw and "hub_height_mm" not in kw and "rim_images" not in kw
    for name in ("render.png", "render.truth.json", "params.json", "overlay.png", "provenance.json"):
        assert (out_dir / name).exists(), name
    assert not (out_dir / "impeller.step").exists()
    truth = json.loads((out_dir / "render.truth.json").read_text())
    assert truth["params"]["inner_diameter"] == 120.0 and "_arr" not in truth
    prov = json.loads((out_dir / "provenance.json").read_text())
    assert prov["blade_count"]["source"] == "measured"
    out = capsys.readouterr().out
    assert "analyzer given: D2=300; D_eye=180; b2=55, b1=75" in out
    assert "variant realistic" in out and out.rstrip().endswith("PASS")
    assert "info (extrapolated" in out and "\nparameter" in out


def test_pipeline3d_option_wiring(tmp_path, capsys, monkeypatch, fake_render3d):
    rc, _, calls = _run_p3d(tmp_path, monkeypatch, lambda kw: _shrouded_result(SHROUDED_PARAMS_FACTORY()),
                            extra=["--variant", "factory", "--preset", "axial_far", "--no-eye-diameter",
                                   "--no-blade-width", "--outer-diameter-mm", "310", "--give-focal",
                                   "--give-hub-height", "--blade-count-hint", "11",
                                   "--rim-azimuths", "240;300", "--clutter"])
    assert rc == 0
    truth_p, preset, rkw = fake_render3d[0]
    assert preset == "axial_far" and truth_p.inner_diameter == 190.0 and rkw["marker"] is True
    assert [c[1] for c in fake_render3d[1:]] == ["side", "side"]
    assert [c[2]["azimuth_deg"] for c in fake_render3d[1:]] == [240.0, 300.0]
    (kw,) = calls
    assert kw["outer_diameter_mm"] == 310.0 and "eye_diameter_mm" not in kw
    assert "blade_width_mm" not in kw and kw["focal_px"] == 1080.0
    assert "hub_height_mm" not in kw            # factory variant has no explicit hub height
    assert kw["blade_count_hint"] == 11 and len(kw["rim_images"]) == 2 and kw["marker_hint"] == "bright"
    out = capsys.readouterr().out
    assert "D2=310" in out and "D_eye" not in out.split("analyzer given:")[1].splitlines()[0]
    assert "focal_px=1080.0" in out and "2 rim image(s)" in out
    # prior path: nothing given
    rc, _, calls = _run_p3d(tmp_path, monkeypatch, lambda kw: _shrouded_result(cli.realistic_params()),
                            extra=["--no-outer-diameter", "--no-eye-diameter", "--no-blade-width",
                                   "--give-hub-height"])
    assert rc == 0
    assert calls[-1] == {"blade_count_hint": None, "hub_height_mm": 20.0}
    assert "analyzer given: hub_height=20" in capsys.readouterr().out


def test_pipeline3d_exit_codes(tmp_path, capsys, monkeypatch, fake_render3d):
    wrong = dataclasses.replace(cli.realistic_params(), blade_count=9)
    rc, out_dir, _ = _run_p3d(tmp_path, monkeypatch, lambda kw: _shrouded_result(wrong))
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL:" in out and "blade_count mismatch (11 vs 9)" in out
    assert (out_dir / "params.json").exists()
    # invalid analyzer output: clean error, params.json not written
    bad = dataclasses.replace(cli.realistic_params(), hub_diameter=400.0)
    rc, out_dir, _ = _run_p3d(tmp_path, monkeypatch, lambda kw: _shrouded_result(bad))
    assert rc == 1
    assert "error: analyzer produced an invalid parameter set" in _err(capsys)
    assert not (out_dir / "params.json").exists() and (out_dir / "render.png").exists()
    # analyzer failure (e.g. no rim ellipse): clean error
    def boom(kw):
        raise ValueError("no rim ellipse found")

    rc, _, _ = _run_p3d(tmp_path, monkeypatch, boom)
    assert rc == 1 and _err(capsys).strip() == "error: no rim ellipse found"
    # a malformed azimuth list
    with pytest.raises(SystemExit) as e:
        cli.main(["pipeline3d", "--out-dir", str(tmp_path / "z"), "--preset", "nope"])
    assert e.value.code == 2
    rc, _, _ = _run_p3d(tmp_path, monkeypatch, lambda kw: _shrouded_result(cli.realistic_params()),
                        extra=["--rim-azimuths", "240,abc"])
    assert rc == 1 and "--rim-azimuths" in _err(capsys)


def test_render_uses_variant_or_params(tmp_path, capsys, monkeypatch, fake_render3d):
    out = tmp_path / "r" / "x.png"
    assert cli.main(["render", "--out", str(out), "--size", "32", "--clutter"]) == 0
    assert fake_render3d[-1][0].inner_diameter == 120.0 and fake_render3d[-1][2]["clutter"] is True
    assert out.exists() and json.loads(out.with_suffix(".truth.json").read_text())["preset"] == "axial_close"
    assert cli.main(["render", "--out", str(out), "--size", "32", "--variant", "factory"]) == 0
    assert fake_render3d[-1][0].inner_diameter == 190.0
    pj = tmp_path / "p.json"
    dataclasses.replace(cli.realistic_params(), blade_count=7).save(pj)
    assert cli.main(["render", "--out", str(out), "--size", "32", "--params", str(pj), "--variant",
                     "factory"]) == 0
    assert fake_render3d[-1][0].blade_count == 7      # --params wins over --variant
    capsys.readouterr()
    # an invalid params file is a clean error
    dataclasses.replace(DEFAULT_PARAMS, hub_diameter=150.0).save(pj)
    assert cli.main(["render", "--out", str(out), "--params", str(pj)]) == 1
    assert "error:" in _err(capsys)


def test_builtin_variants():
    r = cli.realistic_params()
    r.validate()
    assert r.inner_diameter == 120.0 and r.hub_height == 20.0 and r.r1 < r.r_eye
    f = cli.variant_params("factory")
    f.validate()
    assert f.r1 > f.r_eye                  # no blade root visible through the eye by construction
    assert cli.variant_params("realistic").inner_diameter == 120.0
    with pytest.raises(cli.CliError):
        cli.variant_params("nope")
    assert SHROUDED_PARAMS_FACTORY().inner_diameter == 190.0      # the factory preset itself is untouched
