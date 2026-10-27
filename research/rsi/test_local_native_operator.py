"""Source-only local RSI reliability tests; no fake-game success or native rollout."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from local_native_operator import (
    PIN, URL, DATA_SHA, E5_SEEDS, archived_evidence, checked_episodes,
    checkout_status, doctor, file_lock, paired_comparison, replay_preflight,
    run, sha256, visual_video_preflight, execute,
)

ROOT = Path(__file__).resolve().parents[2]

class LocalNativeOperatorTests(unittest.TestCase):
    def test_archived_real_e3_e4_outputs_recomputed(self):
        a = paired_comparison(ROOT)
        self.assertEqual(a['status'], 'archived_native_episode_reaudit_passed')
        self.assertAlmostEqual(a['paired_e3_ppo_minus_bc'], -298.9375)
        self.assertAlmostEqual(a['paired_e4_weighted_minus_bc'], 81.9375)
        self.assertEqual(a['archived_native_episode_groups']['e4_teacher']['stage_clears'], 0)
        self.assertEqual(a['archived_native_episode_groups']['e4_teacher']['timeouts'], 2)
        self.assertFalse(a['fresh_native_game_executed'])

    def test_missing_checkout_never_claims_native(self):
        d = doctor(ROOT)
        self.assertFalse(d['native_ready'])
        self.assertFalse(d['source']['commit_verified'])
        self.assertFalse(d['fresh_native_game_executed'])
        self.assertIn('taisei_ffi.py', ' '.join(d['missing_canonical_source_files']))

    def test_source_checkout_cannot_be_certified_by_bare_zip_or_wrong_lock(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            root = tmp/'source'; root.mkdir()
            lock = tmp/'UPSTREAM.lock'
            lock.write_text(f'REPO_URL={URL}\nCOMMIT={PIN}\n')
            self.assertEqual(checkout_status(root, lock)['status'], 'blocked')
            self.assertIn('Git metadata', checkout_status(root, lock)['reason'])
            (root/'.git').mkdir()
            lock.write_text('REPO_URL=https://invalid\nCOMMIT=123\n')
            self.assertEqual(checkout_status(root, lock)['status'], 'blocked')

    def test_wrong_origin_even_if_git_valid(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t);root = tmp/'source';root.mkdir()
            subprocess.run(['git','init','-q',str(root)],check=True)
            subprocess.run(['git','-C',str(root),'remote','add','origin','https://invalid.org/fake.git'],check=True)
            lock = tmp/'UPSTREAM.lock';lock.write_text(f'REPO_URL={URL}\nCOMMIT={PIN}\n')
            r = checkout_status(root, lock)
            self.assertEqual(r['status'], 'blocked')
            self.assertIn('origin', r['reason'])
            self.assertFalse(r['commit_verified'])

    def test_archive_modified_hash_is_blocked(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            e3 = root/'archival_e3';e4=root/'archival_e4';e3.mkdir();e4.mkdir()
            (e3/'teacher_demonstrations_12000.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                archived_evidence(root)

    def test_wrong_seed_and_sealed_seed_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'trace.jsonl'
            rows=[{'seed':s,'frames':10,'alive_frames':10,'deaths':0,'bombs_used':0,
                   'boss_damage':0,'cleared':False,'truncated':False,'terminated':True,
                   'digest':999,'logical_frame':10} for s in [42132,43042]]
            file.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            with self.assertRaisesRegex(ValueError, 'sealed'):
                checked_episodes(file,E5_SEEDS)

    def test_empty_original_tsr_is_rejected(self):
        raw=bytes.fromhex('686f6e6fe29da4756d690e80010500000022000000'
                          '28b52ffd200001000000000000'
                          '28b52ffd200001000069')
        self.assertEqual(len(raw),44)
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'replay.tsr';file.write_bytes(raw)
            r=replay_preflight(file)
            self.assertEqual(r['status'],'replay_empty_payload')
            self.assertFalse(r['playable_verified'])
            self.assertFalse(r['visual_inspection_performed'])

    def test_visual_gate_refuses_garbage_or_empty(self):
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'gameplay.mp4';file.write_bytes(b'garbage')
            with self.assertRaisesRegex(ValueError,'nonempty'):
                visual_video_preflight(file,Path(t)/'frames')

    def test_video_decode_fixture_never_becomes_gameplay_evidence(self):
        # Solid-color test media validates FFmpeg extraction, NOT actual Taisei.
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            self.skipTest('ffmpeg/ffprobe unavailable')
        with tempfile.TemporaryDirectory() as t:
            v = Path(t)/'solid_color_fixture.mp4'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi',
                            '-i','color=c=gray:s=64x64:r=8:d=2',
                            '-c:v','mpeg4','-q:v','5',str(v)],check=True,timeout=20)
            r=visual_video_preflight(v,Path(t)/'screens')
            self.assertEqual(len(r['frame_samples']),7)
            self.assertFalse(r['genuine_taisei_rendered_origin_verified'])
            self.assertFalse(r['visual_inspection_performed'])
            self.assertFalse(r['gameplay_learning_verified'])

    def test_native_builder_rejects_missing_source_before_build(self):
        from local_native_operator import native_build_if_needed
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(RuntimeError, 'Exact native source identity gate'):
                native_build_if_needed(ROOT, 10**15, Path(t))

    def test_file_lock_blocks_overlap(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'runner.lock'
            file_lock(p)
            with self.assertRaisesRegex(RuntimeError,'overlapping'):
                file_lock(p)

    def test_command_executor_terminates_timeout(self):
        with self.assertRaises(TimeoutError):
            run(['python','-c','import time;time.sleep(4)'],timeout=.08)

    def test_execute_offline_and_native_gated_no_new_training(self):
        with tempfile.TemporaryDirectory() as t:
            dst=Path(t)/'experiment'
            report=execute(ROOT,dst,attempt_native=True)
            self.assertEqual(report['status'],'blocked_native_source_or_build')
            self.assertTrue(report['offline_report'])
            self.assertIsNone(report['native_report'])
            self.assertFalse(report['new_gameplay_improvement_verified'])
            self.assertTrue((dst/'events.jsonl').is_file())
            self.assertTrue((dst/'report.json').is_file())
            self.assertFalse((dst.parent/'.taisei_rsi_single_run.lock').exists())
            with self.assertRaises(FileExistsError):
                execute(ROOT,dst,attempt_native=True)

if __name__=='__main__':unittest.main()
