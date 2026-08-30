# Project Notes for Codex

## Current Focus

This workspace is mainly used for StarVLA QwenFast/QwenEDL experiments on LIBERO,
with a current research focus on applying evidential deep learning (EDL) to
robot VLA policies.

The active EDL implementation lives around:

- `starVLA/model/framework/VLM4A/QwenEDL.py`
- `starVLA/model/framework/VLM4A/QwenFast.py`
- `examples/LIBERO/train_files/run_libero_train.sh`
- `libero_client.zsh`
- `policy_server.zsh`

## Primary Training Workflow

The main training entrypoint is:

```bash
bash examples/LIBERO/train_files/run_libero_train.sh
```

This script currently configures LIBERO training with:

- `Framework_name=QwenEDL`
- `base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action`
- `config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml`
- `libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA`
- `data_mix=libero_all`
- `run_root_dir=./playground/Checkpoints`
- `run_id=qwen3fast_libero_all_edl_0`

Before changing training behavior, inspect this script and the LIBERO YAML
together, because many experiment settings are overridden from the shell script
rather than only from YAML.

## Primary Inference / Evaluation Workflow

LIBERO evaluation is usually run with two local zsh helpers:

```bash
./policy_server.zsh
./libero_client.zsh
```

`policy_server.zsh` starts the policy server from the `starvla` conda
environment. `libero_client.zsh` runs LIBERO evaluation from the `libero` conda
environment.

Both scripts currently point at:

```bash
/mnt/starVLA/playground/Checkpoints/qwen3fast_libero_all_edl_0/checkpoints/steps_30000_pytorch_model.pt
```

When switching experiments, update `CKPT` consistently in both scripts unless
the task explicitly asks for a server/client mismatch.

## Results and Checkpoints

Training outputs are stored under:

```text
playground/Checkpoints/<run_id>/
```

Evaluation results generated during inference tests are generally stored under
each checkpoint run's `results` directory, for example:

```text
playground/Checkpoints/qwen3fast_libero_all_edl_0/results/
playground/Checkpoints/qwen3fast_libero_all/results/
```

Do not delete or reorganize `playground/Checkpoints` unless explicitly asked.
These directories contain experiment checkpoints, copied training scripts,
wandb metadata, final models, and LIBERO result artifacts.

## Implementation Notes

- Prefer the existing StarVLA framework registry patterns when adding or
  modifying QwenFast/QwenEDL behavior.
- Keep EDL-specific changes scoped to `QwenEDL.py` and its immediate training or
  evaluation call sites unless there is a clear cross-framework reason.
- Be careful with existing local modifications in this workspace. Several
  experiment scripts and framework files may already be edited for active runs.
- Use `rg` for code search and inspect local scripts before assuming the current
  training or evaluation command.
<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 83
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/mnt/lzhproject/Auto-claude-code-research-in-sleep`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/mnt/starVLA/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /mnt/lzhproject/Auto-claude-code-research-in-sleep/tools/install_aris_codex.sh "/mnt/starVLA" --reconcile`
For copied Codex installs, use:
`bash /mnt/lzhproject/Auto-claude-code-research-in-sleep/tools/smart_update_codex.sh --project "/mnt/starVLA"`
<!-- ARIS-CODEX:END -->
