"""Verify raw and compressed full-round recovery on CUDA without training."""
import argparse
import gc
import gzip
import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path


def file_hash(path, compressed=False):
    h = hashlib.sha256()
    size = 0
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--record-root', type=Path, required=True)
    parser.add_argument('--resume-state', type=Path, required=True)
    parser.add_argument('--resume-sha256', required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.headless = True
    args.enable_cameras = True
    assert file_hash(args.resume_state)[0] == args.resume_sha256
    app = AppLauncher(args).app
    try:
        import numpy as np
        import torch
        from .dagger_recovery import restore_latest, save_latest
        from .learning import DaggerSession
        from .lifecycle_cleanup import install_stop_handler, memory_snapshot
        from .spec import OUTPUT, WORKTREE
        from .state_fingerprints import expected_fingerprints, expected_next_batch, session_fingerprints, capture_rng, digest
        from .storage import StorageGuard
        install_stop_handler()
        torch.set_num_threads(2)
        payload = torch.load(args.resume_state, map_location='cpu', weights_only=False)
        assert payload['complete_round_boundary'] and payload['status']['formal_training']
        assert not payload['status']['diagnostic_only']
        assert payload['collection']['seen'] == 96000 and payload['student']['updates'] == 6000
        expected = expected_fingerprints(payload)
        expected_batch = expected_next_batch(payload)
        expected_cursor = payload['cursor']
        assert expected_cursor['next_round'] == 3 and expected_cursor['validation_pending'] == 3
        del payload
        gc.collect()
        session = DaggerSession('cuda:0', 256)
        start = time.monotonic()
        status, cursor = restore_latest(session, args.resume_state, expected_sha256=args.resume_sha256, expected_formal=True)
        raw_seconds = time.monotonic() - start
        raw_checks = {k: v == expected[k] for k, v in session_fingerprints(session).items()}
        assert all(raw_checks.values()) and cursor == expected_cursor
        def check_batch():
            rng = session.reservoir.rng.getstate()
            value = digest(session.reservoir.sample(64))
            session.reservoir.rng.setstate(rng)
            assert value == expected_batch
        check_batch()
        guard = StorageGuard(OUTPUT, WORKTREE / 'legmm_skill/tabletop')
        packed = save_latest(session, status, cursor['next_round'], cursor['next_seed'], cursor['episode'], args.out, guard,
                             validation_pending=cursor['validation_pending'], complete_round_boundary=True, compress=True)
        decoded_hash, decoded_bytes = file_hash(args.out / 'latest.pt', compressed=True)
        assert decoded_hash == packed['uncompressed_sha256'] and decoded_bytes == packed['uncompressed_bytes']
        assert packed['encoding'] == 'gzip' and packed['bytes'] < packed['uncompressed_bytes']
        with torch.no_grad():
            next(session.student.model.parameters()).add_(1)
            next(iter(session.student.optimizer.state.values()))['exp_avg'].add_(1)
        session.reservoir.rng.random()
        random.random()
        np.random.random()
        torch.rand(1)
        torch.rand(1, device='cuda:0')
        session.seen += 1
        changed = session_fingerprints(session)
        assert all(changed[k] != expected[k] for k in ('model', 'Adam', 'reservoir', 'global_rng', 'collection'))
        start = time.monotonic()
        restored_status, restored_cursor = restore_latest(session, args.out / 'latest.pt', expected_sha256=packed['sha256'], expected_formal=True)
        compressed_seconds = time.monotonic() - start
        compressed_checks = {k: v == expected[k] for k, v in session_fingerprints(session).items()}
        assert all(compressed_checks.values()) and restored_cursor == expected_cursor
        check_batch()
        rng = capture_rng(True)
        pr = random.Random(); pr.setstate(rng['python'])
        nr = np.random.RandomState(); nr.set_state(rng['numpy'])
        tc = torch.Generator(device='cpu'); tc.set_state(rng['torch'])
        tg = torch.Generator(device='cuda:0'); tg.set_state(rng['cuda'][0])
        random_checks = dict(python=pr.random() == random.random(), numpy=float(nr.random_sample()) == float(np.random.random()),
                             torch_cpu=torch.equal(torch.rand(8, generator=tc), torch.rand(8)),
                             torch_cuda=torch.equal(torch.rand(8, device='cuda:0', generator=tg), torch.rand(8, device='cuda:0')))
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state_all(rng['cuda'])
        assert all(random_checks.values())
        assert session.seen == 96000 and session.student.updates == 6000
        assert file_hash(args.resume_state)[0] == args.resume_sha256
        result = dict(PID=os.getpid(), completed=True, diagnostic_only=True, formal_training=False, phase='compressed_recovery_verified',
                      source_path=str(args.resume_state), source_sha256=args.resume_sha256, original_source_preserved=True,
                      env_decisions=96000, optimizer_updates=6000, env_decisions_added=0, optimizer_updates_added=0, physics_steps=0, scene_created=False,
                      raw_checks=raw_checks, compressed_checks=compressed_checks, next_batch64_exact=True, random_checks=random_checks,
                      deliberate_mutation_replaced=True, cursor=restored_cursor, raw_restore_seconds=raw_seconds,
                      compressed_restore_seconds=compressed_seconds, save_record=packed, resources_after=guard.check(),
                      process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, memory_after=memory_snapshot())
        for destination in (args.out / 'training_status.json', args.record_root / 'round3_compression_GPU_proof.json'):
            destination.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
