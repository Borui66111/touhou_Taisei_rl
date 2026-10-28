#!/usr/bin/env python3
"""Verify the *real* rendered Taisei asset root before launching a skill policy.

No installation, copying or downloading of game assets is performed.  The
compiler/VS environment warning is unrelated to an absent required atlas.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile

REQUIRED_ATLAS = 'portraits.tsatlas.zst'


def inspect_resources(resource_dir: Path) -> dict:
    """Verify the required atlas exists directly or in a Taisei ZIP VFS package.

    Does not establish complete asset availability; it specifically diagnoses
    the known fatal atlas from the provided screenshot.
    """
    root = Path(resource_dir).resolve()
    record = {'path': str(root), 'exists': root.is_dir(), 'atlas_present': False,
              'atlas_location': None, 'status': 'directory_missing'}
    if not root.is_dir():
        return record
    record['status'] = 'atlas_missing'
    plain = root / REQUIRED_ATLAS
    if plain.is_file() and plain.stat().st_size > 0:
        record.update(status='atlas_found', atlas_present=True,
                      atlas_location=str(plain), asset_mode='loose')
        return record
    # ZIP can be a virtual filesystem package; don't decompress or invent assets.
    for archive in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not archive.is_file() or archive.suffix.lower() != '.zip':
            continue
        try:
            with zipfile.ZipFile(archive) as z:
                info = z.getinfo(REQUIRED_ATLAS)
                if info.file_size > 0:
                    record.update(status='atlas_found', atlas_present=True,
                                  atlas_location=f'{archive}!/{REQUIRED_ATLAS}',
                                  asset_mode='zip')
                    return record
        except (OSError, zipfile.BadZipFile, KeyError, ValueError):
            continue
    return record


def candidate_directories(workbench: Path, configured: Path) -> list[Path]:
    """Bounded, known build outputs only: no global filesystem traversal."""
    root = Path(workbench).resolve()
    possibles = [configured,
        root / 'rl_skill_v2' / 'resources',
        root / 'rl_skill_v2' / 'src' / 'resources',
        root / 'rl_skill_v2' / 'data',
        root / 'rl_skill_v2',
        root / 'windows-build' / 'wlive_agent_demo' / 'resources',
        root / 'windows-build' / 'wlive_agent_demo' / 'build' / 'resources',
        root / 'windows-build' / 'wlive_agent_demo' / 'data',
        root / 'windows-build' / 'wlive_agent_demo',
        root / 'windows-build' / 'resources',
        root / 'resources']
    seen, result = set(), []
    for p in possibles:
        p = Path(p).resolve()
        if p not in seen:
            seen.add(p); result.append(p)
    return result


def select_resources(workbench: Path, configured: Path,
                     override: Path | None = None) -> tuple[Path, dict]:
    """Preserve existing valid config; resolve exactly one alternate if needed.

    If multiple alternates exist, require an explicit choice rather than
    silently mixing assets from incompatible game builds.
    """
    if override is not None:
        report = inspect_resources(override)
        if not report['atlas_present']:
            raise FileNotFoundError(
                f"Explicit --resources directory lacks {REQUIRED_ATLAS}: {report['path']}. "
                "Choose the *compiled* resource root for the same Taisei build.")
        return Path(report['path']), {'selected_by': 'explicit_override',
                                      'selected': report, 'candidates': [report]}

    reports = [inspect_resources(p) for p in candidate_directories(workbench, configured)]
    configured_dir = Path(configured).resolve()
    if reports[0]['path'] == str(configured_dir) and reports[0]['atlas_present']:
        return configured_dir, {'selected_by': 'configured_valid',
                                'selected': reports[0], 'candidates': reports}
    good = [r for r in reports if r['atlas_present']]
    if len(good) == 1:
        return Path(good[0]['path']), {'selected_by': 'one_known_alternative',
                                      'selected': good[0], 'candidates': reports}
    details = '\n'.join(
        f"  - {r['path']}: {r['status']}" for r in reports)
    if not good:
        raise FileNotFoundError(
            f"Required Taisei atlas {REQUIRED_ATLAS} was not found in any known "
            f"build resource directory.\n{details}\n"
            "This is an incomplete/misdirected game resource build, not a model failure. "
            "Locate a matching compiled resource root or rebuild this game's "
            "resource targets, then pass --resources PATH. Do not copy random "
            "assets from another Taisei release.")
    raise ValueError(
        f"Several resource roots contain {REQUIRED_ATLAS}; refusing to guess which "
        "matches the executable. Use --resources PATH:\n" +
        '\n'.join('  - '+r['path'] for r in good))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbench', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--configured', type=Path)
    p.add_argument('--resources', type=Path, help='override the resource root to inspect')
    a = p.parse_args(argv)
    root = a.workbench.resolve()
    configured = a.configured or root / 'windows-build' / 'wlive_agent_demo' / 'resources'
    try:
        selected, report = select_resources(root, configured, a.resources)
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'blocked_missing_or_ambiguous_resources',
                          'reason': str(exc), 'asset': REQUIRED_ATLAS}, indent=2))
        return 2
    print(json.dumps({'status': 'preflight_atlas_found', 'resource_root': str(selected),
                      'warning': 'Not a full resource-set or native-game verification',
                      **report}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
