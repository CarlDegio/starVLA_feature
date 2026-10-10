#!/usr/bin/env python3
"""Use the existing VLA trainer, configuring its existing EDL loss controls."""

import argparse
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def configure_wandb_auth():
    """Load the local, git-ignored key without copying it into run configuration."""
    if os.environ.get("WANDB_API_KEY"):
        return
    key_file = Path(os.environ.get("WANDB_API_KEY_FILE", str(Path(__file__).resolve().parents[3] / ".wandb_api_key")))
    if not key_file.is_file():
        if os.environ.get("WANDB_API_KEY_FILE"):
            raise FileNotFoundError("WANDB_API_KEY_FILE does not point to a readable key file")
        return  # Allow an existing wandb login when no local key file is present.
    key = key_file.read_text().strip()
    if not key or len(key.split()) != 1:
        raise ValueError("The wandb key file must contain a single nonempty key")
    os.environ["WANDB_API_KEY"] = key


def configure_edl(model, config):
    """Set loss hyperparameters at launch; do not alter QwenEDL implementation or weights."""
    if config.framework.name != "QwenEDL":
        raise ValueError("This training entrypoint requires framework.name=QwenEDL")
    options = config.framework.edl
    for name, cast in (("loss_type", str), ("topk", int), ("kl_weight", float),
                       ("annealing_steps", int), ("evidence_fn", str)):
        setattr(model, f"edl_{name}", cast(options[name]))
    if model.edl_loss_type not in ("mse", "log", "digamma"):
        raise ValueError("Invalid EDL loss_type")
    if model.edl_evidence_fn not in ("softplus", "relu", "exp"):
        raise ValueError("Invalid EDL evidence_fn")
    if model.edl_topk < 1 or not math.isfinite(model.edl_kl_weight) or model.edl_kl_weight < 0:
        raise ValueError("EDL topk must be positive and KL weight must be finite and nonnegative")
    print(f"EDL controls: loss={model.edl_loss_type}, topk={model.edl_topk}, "
          f"kl_weight={model.edl_kl_weight}, annealing_steps={model.edl_annealing_steps}, "
          f"evidence_fn={model.edl_evidence_fn}", flush=True)
    return model


def configure_gradient_accumulation(config):
    """Keep this real-data workflow at one optimizer update per batch."""
    if int(config.trainer.gradient_accumulation_steps) != 1:
        raise ValueError("Real-data training requires gradient_accumulation_steps=1")
    # Override inherited environment settings before the shared trainer imports Accelerator.
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = "1"


def configure_training_memory(model, config):
    """Apply the requested activation checkpointing in this training entrypoint."""
    backbone = model.qwen_vl_interface.model
    if config.trainer.get("gradient_checkpointing", False):
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        backbone.gradient_checkpointing_disable()
    model.train()
    print("Qwen gradient checkpointing:", backbone.is_gradient_checkpointing, flush=True)
    return model


def finalize_real_training(self, original_finalize):
    """Keep requested step checkpoints without duplicating the final weights."""
    if self.config.trainer.get("save_final_model", True):
        return original_finalize(self)
    if self.accelerator.is_main_process:
        import wandb
        wandb.finish()
    self.accelerator.wait_for_everyone()


def main():
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "no_proxy",
                 "WANDB_HTTP_PROXY", "WANDB_HTTPS_PROXY"):
        os.environ.pop(name, None)
    configure_wandb_auth()
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml),
                          OmegaConf.from_dotlist(normalize_dotlist_args(overrides)))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    # The shared trainer constructs Accelerator at import time. Read the per-run value first.
    configure_gradient_accumulation(cfg)
    from starVLA.training import train_starvla as trainer

    if trainer.accelerator.gradient_accumulation_steps != 1:
        raise ValueError("Accelerator must use gradient_accumulation_steps=1")
    plugin = trainer.accelerator.state.deepspeed_plugin
    if plugin is not None and plugin.deepspeed_config.get("gradient_accumulation_steps") not in (1, "auto"):
        raise ValueError("DeepSpeed must use gradient_accumulation_steps=1 (or auto)")

    build = trainer.build_framework

    def build_with_edl_controls(config):
        model = configure_training_memory(configure_edl(build(config), config), config)
        print("Qwen attention backend:", model.qwen_vl_interface.model.config._attn_implementation, flush=True)
        return model

    # Restrict the factory adapter to this entrypoint. The original trainer/model stay unchanged.
    trainer.build_framework = build_with_edl_controls
    evaluate = trainer.VLATrainer.eval_action_model

    def evaluate_in_eval_mode(self, step_metrics=None):
        # Generation uses its cache in eval mode; restore checkpointed training afterwards.
        was_training = self.model.training
        self.model.eval()
        try:
            return evaluate(self, step_metrics)
        finally:
            self.model.train(was_training)

    trainer.VLATrainer.eval_action_model = evaluate_in_eval_mode
    finalize = trainer.VLATrainer._finalize_training
    trainer.VLATrainer._finalize_training = lambda self: finalize_real_training(self, finalize)
    trainer.main(cfg)


if __name__ == "__main__":
    main()
