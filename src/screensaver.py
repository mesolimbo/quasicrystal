"""Generate a slowly refreshing Ammann–Beenker maze animation for 540×960 e-paper.

The routine below builds an Ammann–Beenker tiling patch via the standard
cut-and-project construction, stitches its vertex set into a single
randomised Hamiltonian-style cycle using a nearest-neighbour/2-opt tour,
and renders the incremental drawing to a 16-level grayscale GIF sized for
an M5 PaperS3 panel. Each frame is delayed by ~10 seconds by default so the
loop animates at an e-paper friendly cadence.
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

from PIL import Image, ImageDraw

WIDTH = 540
HEIGHT = 960
GRAY_LEVELS = 16
MARGIN = 12  # pixel padding around the drawing region


def gray_level(level: int) -> int:
    """Return a legal 4-bit grayscale value mapped to the 0–255 range."""
    level = max(0, min(GRAY_LEVELS - 1, level))
    return int(round(level * 255 / (GRAY_LEVELS - 1)))


def build_palette_image() -> Image.Image:
    """Return a 16-color grayscale palette for quantising GIF frames."""
    palette = []
    for level in range(GRAY_LEVELS):
        value = gray_level(level)
        palette.extend([value, value, value])
    palette.extend([0] * (768 - len(palette)))
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(palette)
    return palette_image


PALETTE_IMAGE = build_palette_image()


def generate_ab_patch(limit: int, window: float, rng: random.Random) -> List[Tuple[float, float]]:
    """Generate vertex coordinates for an Ammann–Beenker tiling patch."""
    angles = [index * math.pi / 4 for index in range(4)]
    physical = [(math.cos(angle), math.sin(angle)) for angle in angles]
    perpendicular = [
        (math.cos(angle + math.pi / 2), math.sin(angle + math.pi / 2)) for angle in angles
    ]

    coords: List[Tuple[float, float]] = []
    for vector in itertools.product(range(-limit, limit + 1), repeat=4):
        x = y = u = v = 0.0
        for coefficient, (px, py), (qx, qy) in zip(vector, physical, perpendicular):
            x += coefficient * px
            y += coefficient * py
            u += coefficient * qx
            v += coefficient * qy
        if max(abs(u), abs(v)) <= window:
            coords.append((x, y))

    if not coords:
        raise ValueError("No tiling vertices were generated; adjust --limit/--window")

    # Apply a random dihedral-8 symmetry to keep runs varied.
    rotation = rng.randrange(8)
    sin_theta = math.sin(rotation * math.pi / 4)
    cos_theta = math.cos(rotation * math.pi / 4)
    flipped = rng.choice([False, True])

    transformed: List[Tuple[float, float]] = []
    for x, y in coords:
        if flipped:
            x = -x
        tx = x * cos_theta - y * sin_theta
        ty = x * sin_theta + y * cos_theta
        transformed.append((tx, ty))

    xs = [x for x, _ in transformed]
    ys = [y for _, y in transformed]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x == 0 or span_y == 0:
        raise ValueError("Degenerate patch dimensions detected")

    scale_x = (WIDTH - 2 * MARGIN) / span_x
    scale_y = (HEIGHT - 2 * MARGIN) / span_y
    scale = min(scale_x, scale_y)
    offset_x = (WIDTH - scale * span_x) / 2 - scale * min_x
    offset_y = (HEIGHT - scale * span_y) / 2 - scale * min_y

    return [(scale * x + offset_x, scale * y + offset_y) for x, y in transformed]


def nearest_neighbour_cycle(points: Sequence[Tuple[float, float]], rng: random.Random) -> List[int]:
    """Generate an initial tour using a nearest-neighbour walk."""
    unused = list(range(len(points)))
    current = unused.pop(rng.randrange(len(unused)))
    cycle = [current]

    while unused:
        cx, cy = points[current]
        next_index = min(
            unused,
            key=lambda idx: (points[idx][0] - cx) ** 2 + (points[idx][1] - cy) ** 2,
        )
        unused.remove(next_index)
        cycle.append(next_index)
        current = next_index

    return cycle


def two_opt_improvement(
    cycle: List[int],
    points: Sequence[Tuple[float, float]],
    rng: random.Random,
    max_rounds: int,
) -> None:
    """Perform a bounded number of 2-opt refinement rounds on the tour."""
    if len(cycle) < 4:
        return

    n = len(cycle)
    for _ in range(max_rounds):
        improved = False
        indices = list(range(n - 2))
        rng.shuffle(indices)
        for i in indices:
            a = cycle[i]
            b = cycle[i + 1]
            ax, ay = points[a]
            bx, by = points[b]
            for j in range(i + 2, n if i > 0 else n - 1):
                c = cycle[j]
                d = cycle[(j + 1) % n]
                if a == d:
                    continue
                cx, cy = points[c]
                dx, dy = points[d]
                current_length = math.hypot(ax - bx, ay - by) + math.hypot(cx - dx, cy - dy)
                swapped_length = math.hypot(ax - cx, ay - cy) + math.hypot(bx - dx, by - dy)
                if swapped_length + 1e-9 < current_length:
                    cycle[i + 1 : j + 1] = reversed(cycle[i + 1 : j + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break


def build_cycle(
    points: Sequence[Tuple[float, float]],
    rng: random.Random,
    two_opt_rounds: int,
) -> List[int]:
    """Construct a Hamiltonian-style cycle visiting each vertex exactly once."""
    cycle = nearest_neighbour_cycle(points, rng)
    two_opt_improvement(cycle, points, rng, two_opt_rounds)
    return cycle


def iter_batches(total: int, batch_size: int) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) slice indices covering range(total) in batches."""
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        yield start, end


def quantise_frames(frames: Sequence[Image.Image]) -> List[Image.Image]:
    """Convert grayscale frames to a shared 16-level palette."""
    return [frame.quantize(palette=PALETTE_IMAGE) for frame in frames]


def render_animation(
    points: Sequence[Tuple[float, float]],
    cycle: Sequence[int],
    sleep_seconds: float,
    max_refreshes: int,
    segments_per_refresh: int | None,
    line_width: int,
    ink_level: int,
    background_level: int,
    gif_path: Path,
    frames_dir: Path | None,
    dry_run: bool,
) -> List[Image.Image]:
    """Draw the curve incrementally and return the rendered frames."""
    extended_cycle = list(cycle) + [cycle[0]]
    total_segments = len(extended_cycle) - 1
    batches = max(1, max_refreshes)
    if segments_per_refresh is None:
        segments_per_refresh = max(1, math.ceil(total_segments / batches))
    else:
        segments_per_refresh = max(1, segments_per_refresh)

    base_image = Image.new("L", (WIDTH, HEIGHT), color=gray_level(background_level))
    draw = ImageDraw.Draw(base_image)
    frames: List[Image.Image] = []

    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, (start, end) in enumerate(iter_batches(total_segments, segments_per_refresh)):
        for segment_index in range(start, end):
            a = extended_cycle[segment_index]
            b = extended_cycle[segment_index + 1]
            draw.line((*points[a], *points[b]), fill=gray_level(ink_level), width=line_width)
        frame = base_image.copy()
        frames.append(frame)
        if frames_dir is not None:
            frame_path = frames_dir / f"frame_{batch_index:04d}.bmp"
            frame.save(frame_path)
        if not dry_run and (batch_index < (total_segments - 1) // segments_per_refresh):
            time.sleep(sleep_seconds)

    if not frames:
        frames.append(base_image)

    quantised = quantise_frames(frames)
    quantised[0].save(
        gif_path,
        save_all=True,
        append_images=quantised[1:],
        loop=0,
        duration=int(max(0.1, sleep_seconds) * 1000),
        disposal=2,
    )
    return frames


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=7, help="Cut-and-project search radius (default: 7)")
    parser.add_argument(
        "--window",
        type=float,
        default=1.6,
        help="Acceptance window radius in perpendicular space (default: 1.6)",
    )
    parser.add_argument(
        "--two-opt-rounds",
        type=int,
        default=4,
        help="Maximum number of 2-opt refinement sweeps applied to the tour",
    )
    parser.add_argument(
        "--max-refreshes",
        type=int,
        default=96,
        help="Maximum number of screen refreshes to draw the entire loop",
    )
    parser.add_argument(
        "--segments-per-refresh",
        type=int,
        default=None,
        help="Override the number of curve segments rendered per refresh",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=10.0,
        help="Seconds to pause between refreshes when not running with --dry-run",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Line thickness in pixels for drawing the path",
    )
    parser.add_argument(
        "--ink-level",
        type=int,
        default=2,
        help="Grayscale level (0-15) used for the maze path",
    )
    parser.add_argument(
        "--background-level",
        type=int,
        default=15,
        help="Grayscale level (0-15) used for the background",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=Path("maze_animation.gif"),
        help="Output GIF path",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Optional directory for dumping intermediate BMP frames",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the real-time sleeps (useful when previewing on a desktop)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    points = generate_ab_patch(limit=args.limit, window=args.window, rng=rng)
    cycle = build_cycle(points, rng=rng, two_opt_rounds=args.two_opt_rounds)
    render_animation(
        points=points,
        cycle=cycle,
        sleep_seconds=args.sleep,
        max_refreshes=args.max_refreshes,
        segments_per_refresh=args.segments_per_refresh,
        line_width=args.line_width,
        ink_level=args.ink_level,
        background_level=args.background_level,
        gif_path=args.gif,
        frames_dir=args.frames_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
