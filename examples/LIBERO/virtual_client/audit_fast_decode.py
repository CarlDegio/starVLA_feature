"""Audit unmodified real inference and teacher-forced FAST tokens without recovery.

Invalid chunks are recorded as errors, never padded/truncated or scored as actions.
Uses the real server adapter and virtual client's dataset observations in-process.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from deployment.real.infer import load_policy, TASKS, ROOT
from examples.LIBERO.virtual_client.client import Episode


def token_audit(tokenizer, tokens, horizon, dimension):
    raw = tokenizer.bpe_tokenizer.decode(tokens)
    return {"token_count": len(tokens), "coefficient_count": len(raw),
            "expected_coefficients": horizon * dimension,
            "replacement_characters": raw.count('\ufffd'),
            "strictly_decodable": len(raw) == horizon * dimension and '\ufffd' not in raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--task', choices=TASKS, default='slippers')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--max-chunks', type=int)
    parser.add_argument('--teacher-forced-chunks', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error('max-chunks must be positive')
    if args.teacher_forced_chunks < 0:
        parser.error('teacher-forced-chunks cannot be negative')
    args.output.mkdir(parents=True, exist_ok=False)
    server_args = SimpleNamespace(ckpt=args.ckpt, unnorm_key=None, no_bf16=False,
                                 prompt=None, execution_horizon=30, fast_decode_mode='strict')
    policy = load_policy(server_args, args.task)
    framework = policy.wrapper._framework
    interface = framework.qwen_vl_interface
    eos_ids = interface.model.generation_config.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
    tokenizer = framework.action_model.fast_tokenizer
    horizon, dimension = framework.action_horizon, 14
    episode = Episode(ROOT / 'playground/Datasets/edl_real' / TASKS[args.task][0], args.episode)
    original_generate = interface.model.generate
    original_predict = policy.wrapper.predict_action
    captured = {}

    def generate(**kwargs):
        generated = original_generate(**kwargs)
        captured['generated'] = generated
        captured['prompt_ids'] = kwargs['input_ids'].detach().clone()
        return generated

    def predict(*, examples, **kwargs):
        captured['example'] = examples[0]
        return original_predict(examples=examples, **kwargs)

    interface.model.generate = generate
    policy.wrapper.predict_action = predict
    # Load the actual training reader for independent preprocessing comparison.
    from omegaconf import OmegaConf
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    cfg = OmegaConf.load(Path(args.ckpt).resolve().parents[1] / 'config.full.yaml')
    training = get_vla_dataset(cfg.datasets.vla_data).datasets[0]
    entries = []
    try:
        for index, frame in enumerate(range(0, episode.length, args.stride)):
            if args.max_chunks is not None and index >= args.max_chunks:
                break
            captured.clear()
            obs = episode.observation(frame)
            started = time.perf_counter()
            entry = {'frame': frame, 'chunk_index': index}
            try:
                response = policy.infer(obs)
                entry['inference_error'] = None
                np.save(args.output / f'frame_{frame:06d}_actions.npy', response['actions'])
            except ValueError as exc:
                if 'FAST decode rejected:' not in str(exc):
                    raise
                entry['inference_error'] = str(exc)
            generated = captured.pop('generated')
            prompt_ids = captured.pop('prompt_ids')
            example = captured.pop('example')
            suffix = generated.sequences[:, prompt_ids.shape[1]:]
            action_ids = framework._extract_action_token_ids(suffix)[0]
            fast_ids = framework._decode_action_tokens([action_ids])[0] or []
            entry['generated'] = token_audit(tokenizer, fast_ids, horizon, dimension)
            entry['generated'].update({'new_tokens': suffix.shape[1],
                'last_token_id': int(suffix[0, -1]),
                'ended_with_eos': int(suffix[0, -1]) in eos_ids,
                'hit_generation_limit': suffix.shape[1] == int(framework.config.trainer.eval_max_new_tokens),
                'non_action_tokens': [int(t) for t in suffix[0].tolist() if not interface._ACTION_TOKEN_MIN <= t <= interface._ACTION_TOKEN_MAX]})
            # No action tokens should be extracted from the input prompt.
            entry['prompt_action_tokens'] = len(framework._extract_action_token_ids(prompt_ids)[0])
            del generated
            raw = training.get_step_data(args.episode, frame)
            sample = training._pack_sample(training.transforms(raw))
            gt_tokens = tokenizer(sample['action'])[0]
            entry['ground_truth'] = token_audit(tokenizer, gt_tokens, horizon, dimension)
            entry['input_parity'] = {
                'language_equal': sample['lang'] == example['lang'],
                'state_equal': bool(np.array_equal(sample['state'], example['state'])),
                'state_max_abs_difference': float(np.max(np.abs(sample['state'].astype(np.float32) - example['state'].astype(np.float32)))),
                'image_equal': [bool(np.array_equal(np.asarray(a), np.asarray(b))) for a, b in zip(sample['image'], example['image'])],
                'image_max_abs_difference': [int(np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).max()) for a, b in zip(sample['image'], example['image'])]}
            if index < args.teacher_forced_chunks:
                inputs = interface.build_qwenvl_inputs(images=[sample['image']],
                    instructions=framework._build_instructions([sample]),
                    solutions=[framework.map_fast_token_to_vlm_action(gt_tokens)])
                labels = inputs.pop('labels')
                start = (labels[0] != -100).nonzero()[0].item()
                entry['input_parity']['train_prefix_equal_to_inference'] = bool(torch.equal(inputs['input_ids'][:, :start], prompt_ids))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = interface(**inputs, return_dict=True, use_cache=False)
                    predictions = outputs.logits[:, :-1].argmax(dim=-1)
                action_mask = (labels[:, 1:] >= interface._ACTION_TOKEN_MIN) & (labels[:, 1:] <= interface._ACTION_TOKEN_MAX)
                predicted = predictions[action_mask].tolist()
                gt = labels[:, 1:][action_mask].tolist()
                tf_fast = [t - interface._ACTION_TOKEN_MIN for t in predicted if interface._ACTION_TOKEN_MIN <= t <= interface._ACTION_TOKEN_MAX]
                entry['teacher_forced'] = token_audit(tokenizer, tf_fast, horizon, dimension)
                entry['teacher_forced']['action_token_accuracy'] = float(np.mean(np.asarray(predicted) == gt))
                entry['teacher_forced']['all_action_tokens_correct'] = predicted == gt
                entry['teacher_forced']['non_action_predictions'] = len(predicted) - len(tf_fast)
                del outputs, inputs, labels, predictions
            entry['elapsed_s'] = time.perf_counter() - started
            np.savez_compressed(args.output / f'frame_{frame:06d}_tokens.npz',
                generated_ids=suffix.detach().cpu().numpy(), fast_ids=np.asarray(fast_ids),
                gt_fast_ids=np.asarray(gt_tokens))
            entries.append(entry)
            with (args.output / 'chunks.jsonl').open('a') as stream:
                stream.write(json.dumps(entry) + '\n')
            print(json.dumps(entry), flush=True)
            del suffix, prompt_ids
    finally:
        episode.close()
    failed = sum(not row['generated']['strictly_decodable'] for row in entries)
    teacher_rows = [row for row in entries if 'teacher_forced' in row]
    teacher_tokens = sum(row['ground_truth']['token_count'] for row in teacher_rows)
    stride30_rows = [row for row in entries if row['frame'] % 30 == 0]
    stride30_failed = sum(not row['generated']['strictly_decodable'] for row in stride30_rows)
    summary = {'checkpoint': str(Path(args.ckpt).resolve()), 'episode': args.episode,
               'episode_frames': episode.length, 'stride': args.stride, 'chunks': len(entries),
               'failed_chunks': failed, 'failure_rate': failed / len(entries) if entries else None,
               'coefficient_counts': dict(Counter(row['generated']['coefficient_count'] for row in entries)),
               'all_gt_decodable': all(row['ground_truth']['strictly_decodable'] for row in entries),
               'generation_limit_hits': sum(row['generated']['hit_generation_limit'] for row in entries),
               'input_parity': {key: all(row['input_parity'][key] for row in entries) for key in ('language_equal', 'state_equal')},
               'all_images_equal_training_reader': all(all(row['input_parity']['image_equal']) for row in entries),
               'teacher_forced_prefix_comparisons': len(teacher_rows),
               'all_compared_prefixes_equal': all(row['input_parity']['train_prefix_equal_to_inference'] for row in teacher_rows),
               'all_generations_ended_with_eos': all(row['generated']['ended_with_eos'] for row in entries),
               'chunks_with_unexpected_text_tokens': sum(any(token not in eos_ids for token in row['generated']['non_action_tokens']) for row in entries),
               'replacement_character_chunks': sum(row['generated']['replacement_characters'] > 0 for row in entries),
               'teacher_forced_token_accuracy': sum(row['ground_truth']['token_count'] * row['teacher_forced']['action_token_accuracy'] for row in teacher_rows) / teacher_tokens if teacher_tokens else None,
               'teacher_forced_valid_chunks': sum(row['teacher_forced']['strictly_decodable'] for row in teacher_rows),
               'stride30_subset': {'chunks': len(stride30_rows), 'failed_chunks': stride30_failed,
                                   'failure_rate': stride30_failed / len(stride30_rows) if stride30_rows else None},
               'fast_decode_mode': 'strict', 'execution_horizon': 30,
               'invalid_chunks_have_action_output': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
