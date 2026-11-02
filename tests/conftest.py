"""Shared pytest fixtures.

``render_cache``: a disk cache of ``fanscan.cad`` meshes and ``fanscan.render3d`` renders so
the shrouded-mode tests do not rebuild the same CAD model (~15 s each) or re-render the same
image (~1-2 s each) on every run.  Entries are keyed on (params, tessellation tolerance,
fanscan/cad.py source) for meshes and (mesh key, preset, render kwargs, fanscan/render3d.py
source) for renders, so a change to the geometry or the renderer invalidates them.

Location (first that applies):
  1. ``$FANSCAN_RENDER_CACHE``
  2. the session scratch directory below (the analyzer author's development area), when
     its parent exists
  3. pytest's own cache directory (``.pytest_cache/d/fanscan_renders``)
Set ``FANSCAN_RENDER_CACHE=off`` to disable the disk cache (everything is then built in memory
for the session only).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRATCH = Path("/tmp/claude-0/-home-user-fnc-sjk/ae3fa800-23db-5c9b-affd-8179edd5d6bf/scratchpad"
                "/v02/integrate/test_shrouded/render_cache")
MESH_TOL = 0.5


def _params_json(params: Any) -> str:
    d = params.to_dict() if hasattr(params, "to_dict") else dict(params)
    return json.dumps({k: v for k, v in d.items() if k != "meta"}, sort_keys=True, default=str)


def _src_hash(module: Any) -> bytes:
    return hashlib.sha1(Path(module.__file__).read_bytes()).digest()


class RenderCache:
    """Memoised ``mesh(params)`` and ``render(params, preset, **kw)`` with an optional disk store."""

    def __init__(self, root: Path | None):
        self.root = root
        if root is not None:
            (root / "mesh").mkdir(parents=True, exist_ok=True)
        self._mesh: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._render: dict[str, tuple[np.ndarray, dict]] = {}
        self.stats = {"mesh_built": 0, "mesh_loaded": 0, "mesh_seconds": 0.0,
                      "rendered": 0, "render_loaded": 0, "render_seconds": 0.0}

    # -- meshes ------------------------------------------------------------------------
    def mesh_key(self, params: Any) -> str:
        from fanscan import cad

        return hashlib.sha1(_src_hash(cad) + _params_json(params).encode()
                            + f"|{MESH_TOL}".encode()).hexdigest()[:20]

    def mesh(self, params: Any) -> tuple[np.ndarray, np.ndarray]:
        key = self.mesh_key(params)
        if key in self._mesh:
            return self._mesh[key]
        f = None if self.root is None else self.root / "mesh" / f"{key}.npz"
        if f is not None and f.exists():
            z = np.load(f)
            VF = (z["V"], z["F"])
            self.stats["mesh_loaded"] += 1
        else:
            from fanscan import cad

            t0 = time.perf_counter()
            V, F = cad.mesh_from_params(params, MESH_TOL)
            VF = (np.asarray(V, np.float64), np.asarray(F, np.int64))
            self.stats["mesh_built"] += 1
            self.stats["mesh_seconds"] += time.perf_counter() - t0
            if f is not None:
                np.savez_compressed(f, V=VF[0], F=VF[1])
        self._mesh[key] = VF
        return VF

    # -- renders -----------------------------------------------------------------------
    def render(self, params: Any, preset: str, **kw: Any) -> tuple[np.ndarray, dict]:
        """``render3d.render_params(params, preset, mesh=<cached>, **kw)``; the truth dict is
        the JSON-serialisable part only (``render3d.truth_json``: no id image / depth)."""
        from fanscan import render3d

        mkey = self.mesh_key(params)
        key = hashlib.sha1(_src_hash(render3d) + mkey.encode() + preset.encode()
                           + json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:20]
        if key in self._render:
            return self._render[key]
        f_img = None if self.root is None else self.root / f"{key}.npz"
        f_truth = None if self.root is None else self.root / f"{key}.json"
        if f_img is not None and f_img.exists() and f_truth.exists():
            img = np.load(f_img)["img"]
            truth = json.loads(f_truth.read_text())
            self.stats["render_loaded"] += 1
        else:
            t0 = time.perf_counter()
            img, truth = render3d.render_params(params, preset, mesh=self.mesh(params), **kw)
            truth = render3d.truth_json(truth)
            self.stats["rendered"] += 1
            self.stats["render_seconds"] += time.perf_counter() - t0
            if f_img is not None:
                np.savez_compressed(f_img, img=img)
                f_truth.write_text(json.dumps(truth))
        self._render[key] = (img, truth)
        return img, truth


def _cache_root(request) -> Path | None:
    env = os.environ.get("FANSCAN_RENDER_CACHE")
    if env:
        return None if env.lower() in ("off", "0", "none") else Path(env)
    if _SCRATCH.parent.parent.exists():
        return _SCRATCH
    try:
        return Path(request.config.cache.mkdir("fanscan_renders"))
    except Exception:  # pragma: no cover - e.g. ``-p no:cacheprovider``
        return None


@pytest.fixture(scope="session")
def render_cache(request) -> RenderCache:
    rc = RenderCache(_cache_root(request))
    yield rc
    s = rc.stats
    if s["mesh_built"] or s["rendered"]:
        print(f"\n[render_cache] {rc.root}: built {s['mesh_built']} meshes in {s['mesh_seconds']:.0f} s, "
              f"rendered {s['rendered']} images in {s['render_seconds']:.0f} s "
              f"(loaded {s['mesh_loaded']} meshes / {s['render_loaded']} renders from disk)")
