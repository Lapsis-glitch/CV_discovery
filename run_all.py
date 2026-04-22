#!/usr/bin/env python
"""Driver: run all five subsection scripts in-process (single featurisation)
and produce outputs/summary.html."""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_playground.io import featurize
from cv_playground.metrics import reset_metric_store
from cv_playground.utils import (
    base_argparser,
    generate_summary_html,
    setup_logging,
)

SCRIPTS = [
    "scripts/01_linear_cvs.py",
    "scripts/02_nonlinear_features_linear_proj.py",
    "scripts/03_nonlinear_manifold.py",
    "scripts/04_deep_learning_cvs.py",
    "scripts/05_gnn_cvs.py",
]

logger = logging.getLogger(__name__)


def _load_module(path: Path):
    # Prefix the module name so it's a valid Python identifier (script files
    # start with digits). Registering in sys.modules is required — some libs
    # (e.g. torch_geometric's Inspector) look up classes' defining module
    # via sys.modules[...].
    name = "cv_script_" + path.stem
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    setup_logging()
    args = base_argparser("CV Discovery Playground – run all methods").parse_args()
    root = Path(__file__).resolve().parent

    reset_metric_store(Path(args.output_dir) / "metrics.json")

    logger.info("Featurising trajectory once (top=%s  traj=%s  stride=%d)",
                args.top, args.traj, args.stride)
    t0 = time.perf_counter()
    feat = featurize(args.top, args.traj, args.stride, args.selection, native=args.native)
    logger.info("Featurisation finished in %.1f s (%d frames, %d atoms)",
                time.perf_counter() - t0, len(feat.frame_indices), feat.n_atoms)

    wall_start = time.perf_counter()
    for script_rel in SCRIPTS:
        script = root / script_rel
        if not script.exists():
            logger.error("Script not found: %s", script)
            continue

        logger.info("=" * 60)
        logger.info("Running %s", script_rel)
        logger.info("=" * 60)

        t0 = time.perf_counter()
        try:
            mod = _load_module(script)
            if not hasattr(mod, "run"):
                logger.error("%s exposes no run(args, feat) entrypoint.", script_rel)
                continue
            mod.run(args, feat=feat)
            status = "OK"
        except Exception:
            status = "FAILED"
            logger.error("✘ %s FAILED:\n%s", script_rel, traceback.format_exc())
        elapsed = time.perf_counter() - t0
        logger.info("%s  [%s, %.1f s]", script_rel, status, elapsed)

    total = time.perf_counter() - wall_start
    logger.info("All scripts finished in %.1f s", total)

    output_root = Path(args.output_dir)
    if output_root.exists():
        html = generate_summary_html(output_root)
        logger.info("Open %s in a browser to compare results.", html)
    else:
        logger.warning(
            "Output directory %s does not exist – no summary generated.", output_root
        )


if __name__ == "__main__":
    main()
