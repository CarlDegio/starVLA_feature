"""Serve StarVLA for YAM, or test recorded observations without robot control."""
import argparse
import json
import logging
import os
from pathlib import Path
import time

import numpy as np

from deployment.model_server.tools import msgpack_numpy
from deployment.real.yam_policy import YAMStarVLAPolicy

ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "blocks": ("classification_the_blocks", "sort_blocks.npz",
               "Collect the blocks on the table by color: place the gray blocks in the left basket and the pink blocks in the right basket."),
    "tubes": ("insert_the_two_tubes_into_the_rack_one_by_one", "insert_tubes.npz",
              "Insert the test tube into the test tube rack."),
    "slippers": ("place_the_slippers_on_the_shoe_rack", "place_slippers.npz",
                 "Place the slippers on the shoe rack."),
}


def checkpoint_for(task):
    run = f"qwen3fast_edl_real_{TASKS[task][0]}_edl_1e-2_cleaned_20261002"
    return ROOT / "playground/Checkpoints" / run / "final_model/pytorch_model.pt"


def load_sample(path, prompt):
    with np.load(path, allow_pickle=False) as data:
        obs = {"state": data["state"], "prompt": prompt or str(data["prompt"].item()),
               "images": {role: data[role] for role in ("top", "left", "right")}}
        gt = data["gt_actions"] if "gt_actions" in data else None
    return obs, gt


class RecordedClient:
    """Bounded connection and response waits; never imports a robot driver."""
    def __init__(self, host, port, timeout):
        from websockets.sync.client import connect
        self.timeout = timeout
        self.socket = connect(f"ws://{host}:{port}", compression=None, max_size=None,
                              proxy=None, open_timeout=10)
        try:
            self.metadata = msgpack_numpy.unpackb(self.socket.recv(timeout=timeout))
            if (self.metadata.get("backend") != "starvla"
                    or self.metadata.get("state_dim") != 14
                    or self.metadata.get("action_dim") != 14
                    or not isinstance(self.metadata.get("horizon"), int)
                    or self.metadata["horizon"] < 1):
                raise ValueError(f"Not a StarVLA YAM server: {self.metadata}")
        except Exception:
            self.socket.close()
            raise

    def infer(self, obs):
        self.socket.send(msgpack_numpy.packb(obs))
        response = self.socket.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(response)
        return msgpack_numpy.unpackb(response)

    def close(self):
        self.socket.close()


def load_policy(args, task):
    from starVLA.model.framework.share_tools import read_mode_config
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    ckpt = Path(args.ckpt or checkpoint_for(task)).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    cfg, _ = read_mode_config(str(ckpt))
    data = cfg["datasets"]["vla_data"]
    model = cfg["framework"]["action_model"]
    if (cfg["framework"]["name"] not in ("QwenEDL", "QwenFast")
            # Compact config.yaml omits state_dim / action_type / include_state.
            # The task's registered real data layout supplies these defaults.
            or model["action_dim"] != 14 or model.get("state_dim", 14) != 14
            or data.get("include_state", False)
            or data.get("action_mode") != "abs"
            or data.get("action_type", "joint_targets") != "joint_targets"
            or data.get("data_mix") != f"edl_real_{TASKS[task][0]}"):
        raise ValueError("Checkpoint must match --task and use the real 14D absolute-joint, image-only configuration")
    wrapper = PolicyServerWrapper(str(ckpt), device="cuda", use_bf16=not args.no_bf16,
                                  unnorm_key=args.unnorm_key)
    policy = YAMStarVLAPolicy(wrapper, task=task, default_prompt=args.prompt or TASKS[task][2])
    policy.metadata["edl_details_available"] = cfg["framework"]["name"] == "QwenEDL"
    policy.metadata["edl"] = cfg["framework"].get("edl", {})
    policy.metadata["edl_uncertainty_definition"] = "action-vocabulary top-k: normalized expected entropy AU; K/sum(alpha) EU"
    return policy


def evaluate(policy, args, inputs, prompt, warmup_s):
    horizon = policy.metadata["horizon"]
    report = {"metadata": policy.metadata, "warmup_s": warmup_s,
              "robot_commands_sent": False, "samples": []}
    for index, path in enumerate(inputs):
        obs, gt = load_sample(path, prompt)
        predictions, latencies = [], []
        for repeat in range(args.repeat):
            started = time.perf_counter()
            result = policy.infer(obs)
            latencies.append(time.perf_counter() - started)
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (horizon, 14) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid actions: {actions.shape}")
            predictions.append(actions)
            np.savez_compressed(args.output / f"{index}_{path.stem}_{repeat}_diagnostics.npz",
                                **result.get("diagnostics", {}))
        stacked = np.stack(predictions)
        np.save(args.output / f"{index}_{path.stem}_actions.npy", stacked)
        entry = {"input": str(path), "prompt": obs["prompt"], "shape": list(stacked.shape),
                 "latency_s": latencies, "finite": True}
        if gt is not None:
            if gt.ndim != 2 or gt.shape[1] != 14 or not np.isfinite(gt).all() or len(gt) == 0:
                raise ValueError("Recorded gt_actions must be finite (T, 14) in YAM order")
            count = min(horizon, len(gt))
            error = np.abs(stacked[:, :count] - gt[None, :count])
            entry.update({"compared_steps": count,
                          "joint_mae_rad": float(error[..., [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].mean()),
                          "gripper_mae": float(error[..., [6, 13]].mean())})
        report["samples"].append(entry)
        print(json.dumps(entry), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("serve", "offline", "client"), default="serve")
    parser.add_argument("--task", choices=TASKS, default=os.environ.get("TASK"))
    parser.add_argument("--ckpt", default=os.environ.get("CKPT"))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8002")))
    parser.add_argument("--unnorm-key")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--samples-root", type=Path, default=Path(os.environ.get(
        "YAM_SAMPLES_ROOT", str(ROOT.parent / "starvla-yam-inference/samples"))))
    parser.add_argument("--input", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", help="Override the real-data converter's task prompt")
    parser.add_argument("--use-sample-prompt", action="store_true",
                        help="Use the NPZ prompt verbatim instead of the task default")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Serve without a recorded warmup sample")
    args = parser.parse_args()
    if args.repeat < 1 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("repeat and timeout must be positive")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt cannot be empty")
    if args.prompt is not None and args.use_sample_prompt:
        parser.error("--prompt and --use-sample-prompt are mutually exclusive")
    if args.mode != "serve" and args.output is None:
        parser.error("--output is required for offline/client")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    policy = None
    try:
        if args.mode == "client":
            policy = RecordedClient(args.host, args.port, args.timeout)
            task = policy.metadata["task"]
            if args.task and args.task != task:
                raise ValueError(f"Requested task {args.task} but server task is {task}")
        else:
            task = args.task or "blocks"
        inputs = args.input or [args.samples_root / TASKS[task][1]]
        if args.mode != "serve" or not args.skip_warmup:
            for path in inputs:
                if not path.is_file():
                    raise FileNotFoundError(f"Sample not found: {path}; supply --input or serve --skip-warmup")
        if policy is None:
            policy = load_policy(args, task)
        prompt = None if args.use_sample_prompt else (args.prompt or policy.metadata["default_prompt"])
        warmup_s = None
        if not args.skip_warmup:
            started = time.perf_counter()
            policy.infer(load_sample(inputs[0], prompt)[0])
            warmup_s = time.perf_counter() - started
        print(json.dumps({"metadata": policy.metadata, "warmup_s": warmup_s}), flush=True)
        if args.mode == "serve":
            from deployment.real.websocket_server import YAMWebsocketServer
            YAMWebsocketServer(policy, args.host, args.port).serve_forever()
        else:
            evaluate(policy, args, inputs, prompt, warmup_s)
    finally:
        if isinstance(policy, RecordedClient):
            policy.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
