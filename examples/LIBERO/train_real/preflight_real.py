#!/usr/bin/env python3
"""Check one task configuration before starting an expensive training run."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main():
    from omegaconf import OmegaConf
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--check-data-only", action="store_true", help="Validate configuration/data without requiring the GPU training packages.")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml),
                          OmegaConf.from_dotlist(normalize_dotlist_args(overrides)))
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
    mixture = DATASET_NAMED_MIXTURES[cfg.datasets.vla_data.data_mix]
    if len(mixture) != 1 or mixture[0][2] != "edl_real_dual_arm":
        raise ValueError("Each real-data run must select exactly one edl_real task")
    root = Path(cfg.datasets.vla_data.data_root_dir) / mixture[0][0]
    summary = json.loads((root / "meta/conversion_complete.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    if summary["episodes"] != info["total_episodes"] or summary["frames"] != info["total_frames"]:
        raise ValueError("Dataset completion metadata does not match info.json")
    manifest = json.loads((root / "meta/conversion_manifest.json").read_text())
    if "normalization" in manifest:
        from starVLA.dataloader.gr00t_lerobot.datasets import _load_stats_cache
        cached = _load_stats_cache(root / "meta/stats_gr00t.json", {"mode": "abs"}, invalidate_legacy=False)
        stats = json.loads((root / "meta/stats.json").read_text())
        columns = ["action"]
        if cfg.datasets.vla_data.get("include_state", False):
            columns.append("observation.state")
        if cached is None or any(
                cached.get(column, {}).get(key) != stats[column][key]
                for column in columns for key in ("q01", "q99", "mean")):
            raise ValueError("Real-data normalization cache missing or changed; rerun the conversion command to restore it")
    if cfg.framework.name != "QwenEDL" or cfg.framework.action_model.action_dim != 14:
        raise ValueError("Real datasets require QwenEDL with action_dim=14")
    include_state = cfg.datasets.vla_data.get("include_state", False)
    state_input = cfg.framework.get("state_input", {})
    if bool(include_state) != bool(state_input.get("enabled", False)):
        raise ValueError("include_state and framework.state_input.enabled must be enabled/disabled together")
    if include_state:
        expected_names = [*[f"left_joint_{i}" for i in range(6)],
                          *[f"right_joint_{i}" for i in range(6)], "left_gripper", "right_gripper"]
        feature = info["features"]["observation.state"]
        if (cfg.framework.action_model.state_dim != 14 or feature["shape"] != [14]
                or feature["names"] != expected_names):
            raise ValueError("State must use 14D left joints/right joints/left gripper/right gripper order")
        if int(state_input.get("num_bins", 256)) < 2:
            raise ValueError("State input requires num_bins >= 2")
    from examples.LIBERO.train_real.data_config import ACTION_HORIZON
    if (cfg.framework.action_model.action_horizon != ACTION_HORIZON
            or cfg.framework.action_model.future_action_window_size != ACTION_HORIZON - 1
            or cfg.framework.action_model.past_action_window_size != 0):
        raise ValueError(f"Real-data registry requires action_horizon={ACTION_HORIZON}, future window={ACTION_HORIZON - 1}, past window=0")
    if cfg.datasets.vla_data.action_mode != "abs":
        raise ValueError("The recorded labels and default real-data scheme use absolute joint targets")
    from examples.LIBERO.train_real.train_edl_real import configure_edl, configure_gradient_accumulation
    configure_gradient_accumulation(cfg)
    configure_edl(SimpleNamespace(), cfg)
    if summary["video_size"] != [320, 240]:
        raise ValueError("Expected stored videos at 320x240")
    if not Path(cfg.framework.qwenvl.base_vlm).is_dir():
        raise FileNotFoundError(f"Base VLM not found: {cfg.framework.qwenvl.base_vlm}")
    if not Path("playground/Pretrained_models/fast").is_dir():
        raise FileNotFoundError("FAST processor not found at playground/Pretrained_models/fast")
    output = Path(cfg.run_root_dir) / cfg.run_id
    checkpoint_files = list((output / "checkpoints").glob("*"))
    final_files = list((output / "final_model").glob("*"))
    if (checkpoint_files or final_files) and not cfg.trainer.get("is_resume", False):
        raise FileExistsError(f"Existing model output: {output}; choose RUN_ID or set RESUME=1")
    if not args.check_data_only:
        for dependency in ("accelerate", "deepspeed", "transformers", "wandb"):
            if importlib.util.find_spec(dependency) is None:
                raise ImportError(f"Missing {dependency}; run in your StarVLA training environment")
        if (cfg.framework.qwenvl.attn_implementation == "flash_attention_2"
                and importlib.util.find_spec("flash_attn") is None):
            raise ImportError("Real-data training requests flash_attention_2; install flash-attn in this training environment")
    print(f"Preflight OK: {mixture[0][0]}, {summary['episodes']} episodes, {summary['frames']} frames")


if __name__ == "__main__":
    main()
