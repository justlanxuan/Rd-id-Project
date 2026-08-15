"""Tracklet merging utilities (ported from MotionBERT)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

EPS = 1e-6


@dataclass
class TrackletStats:
    track_id: int
    start: int
    end: int
    length: int
    head_center: np.ndarray  # [2]
    tail_center: np.ndarray  # [2]
    head_size: np.ndarray    # [2] (w, h)
    tail_size: np.ndarray    # [2] (w, h)
    head_vel: np.ndarray     # [2]
    tail_vel: np.ndarray     # [2]


@dataclass
class CandidateEdge:
    src: int
    dst: int
    score: float
    gap: int
    norm_dist: float
    size_diff: float
    vel_residual: float
    gap_penalty: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge fragmented AlphaPose IDs and re-export global IDs."
    )
    parser.add_argument("--json_path", type=str, required=True, help="AlphaPose results JSON path")
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default=None,
        help="Directory containing person_<id>_representation.npy (for MotionBERT mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="merged_embeddings",
        help="Output directory for merged embeddings and mapping files",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output path for merged AlphaPose JSON (for Autism-project mode)",
    )

    parser.add_argument("--tail_window", type=int, default=8, help="Tail window length for stats")
    parser.add_argument("--head_window", type=int, default=8, help="Head window length for stats")

    parser.add_argument(
        "--max_gap",
        type=int,
        default=10000000,
        help="Maximum allowed frame gap for candidate linking",
    )
    parser.add_argument(
        "--known_num_people",
        type=int,
        default=None,
        help="Optional known number of people, stop merging when reached",
    )

    parser.add_argument("--weight_a", type=float, default=1.0, help="Weight for norm_dist")
    parser.add_argument("--weight_b", type=float, default=0.6, help="Weight for size_diff")
    parser.add_argument("--weight_c", type=float, default=0.4, help="Weight for velocity_residual")
    parser.add_argument("--weight_d", type=float, default=0.02, help="Weight for gap_penalty")

    parser.add_argument(
        "--score_thresh",
        type=float,
        default=2.2,
        help="Only candidate edges with score <= score_thresh can be merged",
    )
    parser.add_argument(
        "--max_norm_dist",
        type=float,
        default=2.8,
        help="Hard gate for normalized distance",
    )
    parser.add_argument(
        "--max_size_diff",
        type=float,
        default=1.8,
        help="Hard gate for bbox size difference",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS for gap_penalty")

    parser.add_argument(
        "--fill_gaps",
        action="store_true",
        help="Fill uncovered frames in merged embeddings with linear interpolation",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only compute mapping; do not export",
    )
    return parser.parse_args()


def frame_id_from_image_id(image_id, fallback_idx):
    base = os.path.basename(str(image_id))
    stem = os.path.splitext(base)[0]
    nums = re.findall(r"\d+", stem)
    if not nums:
        return fallback_idx
    return int(nums[-1])


def parse_person_id(raw_idx):
    if isinstance(raw_idx, (list, tuple, np.ndarray)):
        if len(raw_idx) == 0:
            return -1
        raw_idx = raw_idx[0]
    return int(raw_idx)


def load_json_records(json_path: str):
    with open(json_path, "r") as f:
        results = json.load(f)

    records = []

    expanded = []
    for i, item in enumerate(results):
        if isinstance(item, dict) and "result" in item and isinstance(item["result"], list):
            image_id = item.get("imgname", item.get("image_id", i))
            for sub in item["result"]:
                merged = dict(sub)
                merged["image_id"] = image_id
                expanded.append(merged)
        else:
            expanded.append(item)

    for i, item in enumerate(expanded):
        try:
            pid = parse_person_id(item.get("idx", -1))
            frame_id = frame_id_from_image_id(item.get("image_id", i), i)
            box = item.get("box", [0, 0, 0, 0])
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = map(float, box)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            records.append((frame_id, pid, cx, cy, w, h))
        except Exception:
            continue

    if not records:
        raise ValueError("No valid records loaded from JSON.")
    records.sort(key=lambda x: (x[0], x[1]))
    return records


def robust_velocity(ts: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(ts) < 2:
        return np.zeros(2, dtype=np.float32)
    dt = float(ts[-1] - ts[0])
    if dt <= 0:
        return np.zeros(2, dtype=np.float32)
    vx = float(xs[-1] - xs[0]) / dt
    vy = float(ys[-1] - ys[0]) / dt
    return np.array([vx, vy], dtype=np.float32)


def compute_tracklet_stats(records, head_window: int, tail_window: int):
    by_id = defaultdict(list)
    frame_to_ids = defaultdict(set)

    for frame_id, pid, cx, cy, w, h in records:
        by_id[pid].append((frame_id, cx, cy, w, h))
        frame_to_ids[frame_id].add(pid)

    stats = {}
    for pid, items in by_id.items():
        items = sorted(items, key=lambda x: x[0])
        frames = np.array([i[0] for i in items], dtype=np.int64)
        xs = np.array([i[1] for i in items], dtype=np.float32)
        ys = np.array([i[2] for i in items], dtype=np.float32)
        ws = np.array([i[3] for i in items], dtype=np.float32)
        hs = np.array([i[4] for i in items], dtype=np.float32)

        start = int(frames[0])
        end = int(frames[-1])
        length = len(frames)

        head_slice = slice(0, min(head_window, length))
        tail_slice = slice(max(0, length - tail_window), length)

        head_center = np.array([xs[head_slice].mean(), ys[head_slice].mean()], dtype=np.float32)
        tail_center = np.array([xs[tail_slice].mean(), ys[tail_slice].mean()], dtype=np.float32)
        head_size = np.array([ws[head_slice].mean(), hs[head_slice].mean()], dtype=np.float32)
        tail_size = np.array([ws[tail_slice].mean(), hs[tail_slice].mean()], dtype=np.float32)

        head_vel = robust_velocity(frames[head_slice], xs[head_slice], ys[head_slice])
        tail_vel = robust_velocity(frames[tail_slice], xs[tail_slice], ys[tail_slice])

        stats[pid] = TrackletStats(
            track_id=int(pid),
            start=start,
            end=end,
            length=length,
            head_center=head_center,
            tail_center=tail_center,
            head_size=head_size,
            tail_size=tail_size,
            head_vel=head_vel,
            tail_vel=tail_vel,
        )

    return stats, frame_to_ids


def compute_candidate_edges(
    stats: Dict[int, TrackletStats],
    max_gap: int,
    weight_a: float,
    weight_b: float,
    weight_c: float,
    weight_d: float,
    fps: float,
):
    ids = sorted(stats.keys())
    edges = []

    for src in ids:
        for dst in ids:
            if src == dst:
                continue
            s = stats[src]
            t = stats[dst]
            gap = t.start - s.end
            if gap <= 0 or gap > max_gap:
                continue

            dist = np.linalg.norm(t.head_center - s.tail_center)
            size = 0.5 * (s.tail_size + t.head_size)
            size_norm = np.linalg.norm(size) + EPS
            norm_dist = float(dist / size_norm)

            size_diff = float(np.linalg.norm(np.log((t.head_size + EPS) / (s.tail_size + EPS))))

            dt = float(gap) / float(fps)
            pred = s.tail_center + s.tail_vel * dt
            vel_residual = float(np.linalg.norm(t.head_center - pred) / size_norm)

            gap_penalty = math.log(1.0 + gap)

            score = (
                weight_a * norm_dist
                + weight_b * size_diff
                + weight_c * vel_residual
                + weight_d * gap_penalty
            )

            edges.append(
                CandidateEdge(
                    src=src,
                    dst=dst,
                    score=score,
                    gap=gap,
                    norm_dist=norm_dist,
                    size_diff=size_diff,
                    vel_residual=vel_residual,
                    gap_penalty=gap_penalty,
                )
            )

    return edges


def merge_edges(
    stats: Dict[int, TrackletStats],
    edges: List[CandidateEdge],
    score_thresh: float,
    max_norm_dist: float,
    max_size_diff: float,
    known_num_people: int | None,
):
    parent = {k: k for k in stats}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for e in sorted(edges, key=lambda x: x.score):
        if e.score > score_thresh:
            continue
        if e.norm_dist > max_norm_dist or e.size_diff > max_size_diff:
            continue
        union(e.src, e.dst)
        if known_num_people is not None:
            groups = {find(k) for k in stats}
            if len(groups) <= known_num_people:
                break

    merged = defaultdict(set)
    for k in stats:
        merged[find(k)].add(k)
    return merged


def export_embeddings(
    embedding_dir: str,
    output_dir: str,
    merged_groups: Dict[int, Set[int]],
    fill_gaps: bool,
):
    embedding_dir = Path(embedding_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for new_id, members in enumerate(sorted(merged_groups.values(), key=lambda x: sorted(list(x)))):
        all_embeddings = []
        all_frames = []
        for pid in sorted(members):
            emb_path = embedding_dir / f"person_{pid}_representation.npy"
            if not emb_path.exists():
                continue
            emb = np.load(emb_path)
            frame_ids = np.arange(len(emb))
            all_embeddings.append(emb)
            all_frames.append(frame_ids)

        if not all_embeddings:
            continue

        embeddings = np.concatenate(all_embeddings, axis=0)
        frames = np.concatenate(all_frames, axis=0)
        order = np.argsort(frames)
        embeddings = embeddings[order]
        frames = frames[order]

        if fill_gaps:
            full_frames = np.arange(frames.min(), frames.max() + 1)
            filled = np.zeros((len(full_frames), embeddings.shape[1]), dtype=np.float32)
            filled[:] = np.nan
            for i, f in enumerate(frames):
                filled[f - frames.min()] = embeddings[i]

            # simple linear interpolation over NaNs
            for j in range(filled.shape[1]):
                col = filled[:, j]
                mask = ~np.isnan(col)
                if mask.sum() < 2:
                    continue
                filled[:, j] = np.interp(full_frames, full_frames[mask], col[mask])

            embeddings = filled
            frames = full_frames

        out_path = output_dir / f"person_{new_id}_representation.npy"
        np.save(out_path, embeddings)
        mapping[new_id] = sorted([int(x) for x in members])

    map_path = output_dir / "merged_id_mapping.json"
    with map_path.open("w") as f:
        json.dump(mapping, f, indent=2)

    return mapping


def export_merged_json(
    json_path: str,
    merged_groups: Dict[int, Set[int]],
    output_json_path: str,
):
    """Rewrite AlphaPose JSON with merged global IDs."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build old_id -> global_id mapping
    old_to_global = {}
    for global_id, members in enumerate(sorted(merged_groups.values(), key=lambda x: sorted(list(x)))):
        for old_id in members:
            old_to_global[int(old_id)] = global_id

    def rewrite_item(item):
        if isinstance(item, dict):
            raw_idx = item.get("idx", -1)
            pid = parse_person_id(raw_idx)
            new_pid = old_to_global.get(pid, pid)
            item = dict(item)
            item["idx"] = new_pid
        return item

    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, dict) and "result" in item and isinstance(item["result"], list):
                new_item = dict(item)
                new_item["result"] = [rewrite_item(sub) for sub in item["result"]]
                out.append(new_item)
            else:
                out.append(rewrite_item(item))
    else:
        out = rewrite_item(data)

    output_json = Path(output_json_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(out, f, indent=2)

    mapping = {global_id: sorted([int(x) for x in members]) for global_id, members in enumerate(sorted(merged_groups.values(), key=lambda x: sorted(list(x))))}
    map_path = output_json.parent / (output_json.stem + "_id_mapping.json")
    with map_path.open("w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Exported merged JSON: {output_json}")
    print(f"Exported ID mapping: {map_path}")
    return mapping


def main():
    args = parse_args()
    records = load_json_records(args.json_path)
    stats, _ = compute_tracklet_stats(records, args.head_window, args.tail_window)

    edges = compute_candidate_edges(
        stats=stats,
        max_gap=args.max_gap,
        weight_a=args.weight_a,
        weight_b=args.weight_b,
        weight_c=args.weight_c,
        weight_d=args.weight_d,
        fps=args.fps,
    )

    merged_groups = merge_edges(
        stats=stats,
        edges=edges,
        score_thresh=args.score_thresh,
        max_norm_dist=args.max_norm_dist,
        max_size_diff=args.max_size_diff,
        known_num_people=args.known_num_people,
    )

    if args.dry_run:
        print("Merged groups:", merged_groups)
        return

    if args.output_json:
        export_merged_json(args.json_path, merged_groups, args.output_json)
    else:
        if not args.embedding_dir:
            raise ValueError("--embedding_dir is required when --output_json is not set")
        export_embeddings(
            embedding_dir=args.embedding_dir,
            output_dir=args.output_dir,
            merged_groups=merged_groups,
            fill_gaps=args.fill_gaps,
        )


if __name__ == "__main__":
    main()
