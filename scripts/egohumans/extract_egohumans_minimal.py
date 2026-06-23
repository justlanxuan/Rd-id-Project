#!/usr/bin/env python
"""
Minimal extraction of EgoHumans tar.gz archives for MoBInd IMU generation.
Only extracts processed_data/poses2d/cam03 and processed_data/smpl,
which are the only directories required by MoBInd's extract_data.py.
"""
import os
import subprocess
import argparse
from pathlib import Path


def extract_minimal(data_dir, dry_run=False):
    raw_data_dir = os.path.join(data_dir, 'data')
    actions = sorted([d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))])

    for act in actions:
        act_path = os.path.join(raw_data_dir, act)
        archives = sorted([f for f in os.listdir(act_path) if f.endswith('.tar.gz')])

        for archive in archives:
            tar_path = os.path.join(act_path, archive)
            seq_name = archive.replace('.tar.gz', '')
            extracted_folder = os.path.join(act_path, seq_name)

            if os.path.exists(extracted_folder):
                print(f"Skipping {tar_path}, folder already exists.")
                continue

            print(f"Extracting {tar_path} ...")
            if dry_run:
                continue

            # Strip 8 leading path components to remove the absolute prefix in the archive.
            # Include only the two required subdirectories.
            cmd = [
                'tar', '-xzvf', tar_path,
                '--strip-components=8',
                f'--wildcards',
                f'*/{seq_name}/processed_data/poses2d/cam03/*',
                f'*/{seq_name}/processed_data/smpl/*',
            ]
            try:
                subprocess.run(cmd, cwd=act_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"  -> {extracted_folder}")
            except subprocess.CalledProcessError as e:
                print(f"  ERROR extracting {tar_path}: {e.stderr.decode('utf-8', errors='ignore')[:500]}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='/data/lyxie/ReID/Data/egohumans')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    extract_minimal(args.data_dir, dry_run=args.dry_run)
