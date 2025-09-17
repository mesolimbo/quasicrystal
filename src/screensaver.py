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
import logging
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

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


@dataclass
class TilingPatch:
    """Container holding vertex coordinates and edge connectivity for a patch."""

    points: List[Tuple[float, float]]
    edges: List[Tuple[int, int]]


@dataclass
class CycleBias:
    """Metadata describing how to bias the Hamiltonian towards circular motifs."""

    radii: List[float]
    angles: List[float]
    ring_ids: List[int]
    ring_positions: List[float]
    ring_norm_spacing: List[float]
    radius_scale: float
    mean_radius: float
    radial_weight: float
    angle_weight: float
    cross_weight: float
    skip_weight: float
    skip_multiplier: float
    drift_weight: float
    spiral_strength: float
    spiral_frequency: int
    spiral_phase: float
    spiral_direction: int
    direction: int
    ring_counts: List[int]
    remaining_per_ring: List[int]
    ring_width: float
    jitter: float


def compute_cycle_bias(points: Sequence[Tuple[float, float]], rng: random.Random) -> CycleBias | None:
    """Return heuristic information that biases the tour towards concentric loops."""

    if len(points) < 8:
        # Tiny patches don't benefit meaningfully from the additional structure.
        return None

    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    radii = [math.hypot(x - cx, y - cy) for x, y in points]
    if not radii:
        return None

    angles = [math.atan2(y - cy, x - cx) for x, y in points]
    min_radius = min(radii)
    max_radius = max(radii)
    radius_span = max_radius - min_radius
    if radius_span <= 1e-6:
        return None

    # Break the patch into a stack of radial "rings" so that we can
    # encourage local walks that sweep around each band before moving on.
    ring_count = max(4, min(48, int(round(math.sqrt(len(points))))))
    ring_width = radius_span / ring_count if ring_count > 0 else radius_span
    if ring_width <= 1e-6:
        return None

    ring_ids: List[int] = []
    for radius in radii:
        ring_index = int((radius - min_radius) / ring_width)
        ring_ids.append(max(0, min(ring_count - 1, ring_index)))

    ring_counts = [0] * ring_count
    for ring_index in ring_ids:
        ring_counts[ring_index] += 1

    ring_members: List[List[Tuple[float, int]]] = [[] for _ in range(ring_count)]
    for index, (ring_index, angle) in enumerate(zip(ring_ids, angles)):
        ring_members[ring_index].append((angle, index))

    ring_positions = [0.0] * len(points)
    ring_norm_spacing = [1.0] * ring_count
    for ring_index, members in enumerate(ring_members):
        if not members:
            continue
        members.sort(key=lambda pair: pair[0])
        offset = rng.random()
        count = len(members)
        ring_norm_spacing[ring_index] = 1.0 / count
        for order, (_, idx) in enumerate(members):
            ring_positions[idx] = (order + offset) / count

    mean_radius = sum(radii) / len(radii)
    radius_scale = statistics.pstdev(radii)
    if radius_scale <= 1e-6:
        radius_scale = max(mean_radius, 1.0)

    radial_weight = max(16.0, radius_span / 4.0, mean_radius * 0.25)
    angle_weight = max(8.0, radius_span / 6.0)
    cross_weight = angle_weight * 0.6
    skip_weight = angle_weight * 1.4
    skip_multiplier = 2.4
    drift_weight = radial_weight * 0.45
    spiral_strength = max(ring_width, radius_scale / max(1, ring_count)) * rng.uniform(0.85, 1.4)
    spiral_frequency = rng.choice([2, 3, 4, 5])
    spiral_phase = rng.random() * (2 * math.pi)
    spiral_direction = rng.choice([-1, 1])
    direction = rng.choice([-1, 1])
    remaining_per_ring = ring_counts.copy()
    jitter = max(0.05, radius_span / len(points) * 0.4)

    return CycleBias(
        radii=radii,
        angles=angles,
        ring_ids=ring_ids,
        ring_positions=ring_positions,
        ring_norm_spacing=ring_norm_spacing,
        radius_scale=radius_scale,
        mean_radius=mean_radius,
        radial_weight=radial_weight,
        angle_weight=angle_weight,
        cross_weight=cross_weight,
        skip_weight=skip_weight,
        skip_multiplier=skip_multiplier,
        drift_weight=drift_weight,
        spiral_strength=spiral_strength,
        spiral_frequency=spiral_frequency,
        spiral_phase=spiral_phase,
        spiral_direction=spiral_direction,
        direction=direction,
        ring_counts=ring_counts,
        remaining_per_ring=remaining_per_ring,
        ring_width=ring_width,
        jitter=jitter,
    )


def _biased_transition_cost(
    current: int,
    candidate: int,
    points: Sequence[Tuple[float, float]],
    bias: CycleBias,
    rng: random.Random,
) -> float:
    """Return a weighted cost encouraging circular sweeps with gentle radial drift."""

    cx, cy = points[current]
    nx, ny = points[candidate]
    distance = math.hypot(nx - cx, ny - cy)

    r_current = bias.radii[current]
    r_candidate = bias.radii[candidate]
    ring_current = bias.ring_ids[current]
    ring_candidate = bias.ring_ids[candidate]
    ring_gap = abs(ring_candidate - ring_current)

    ring_width = max(bias.ring_width, 1e-6)
    remaining = max(0, bias.remaining_per_ring[ring_current])
    total = max(1, bias.ring_counts[ring_current])
    remaining_ratio = remaining / total
    base_weight = bias.radial_weight * (0.35 + 0.65 * remaining_ratio)
    radial_penalty = base_weight * ((abs(r_candidate - r_current) / ring_width) ** 2)
    if ring_gap > 0:
        radial_penalty *= 1.0 + 0.6 * ring_gap

    pos_current = bias.ring_positions[current]
    pos_candidate = bias.ring_positions[candidate]
    forward = (pos_candidate - pos_current) % 1.0
    backward = (pos_current - pos_candidate) % 1.0
    if bias.direction < 0:
        forward, backward = backward, forward

    if ring_current == ring_candidate:
        target = bias.ring_norm_spacing[ring_current]
        if target <= 0:
            target = 1.0
        angle_penalty = bias.angle_weight * (forward - target) ** 2
        skip_threshold = target * bias.skip_multiplier
        if forward > skip_threshold:
            angle_penalty += bias.skip_weight * (forward - skip_threshold) ** 2
    else:
        alignment = min(forward, backward)
        angle_penalty = bias.cross_weight * alignment ** 2
        if ring_gap > 1:
            angle_penalty *= 1.0 + 0.4 * (ring_gap - 1)

    spiral = math.sin(
        bias.spiral_frequency * pos_current * (2 * math.pi) + bias.spiral_phase
    )
    desired_radius = r_current + bias.spiral_direction * spiral * bias.spiral_strength
    drift_scale = 0.5 if ring_gap == 0 else 1.0 + 0.25 * ring_gap
    drift_penalty = bias.drift_weight * drift_scale * (
        (r_candidate - desired_radius) / (ring_width * 1.5)
    ) ** 2

    return (
        distance
        + radial_penalty
        + angle_penalty
        + drift_penalty
        + bias.jitter * rng.random()
    )


def generate_ab_patch(limit: int, window: float, rng: random.Random) -> TilingPatch:
    """Generate an Ammann–Beenker tiling patch with vertex and edge data."""
    angles = [index * math.pi / 4 for index in range(4)]
    physical = [(math.cos(angle), math.sin(angle)) for angle in angles]
    perpendicular = [
        (math.cos(angle + math.pi / 2), math.sin(angle + math.pi / 2)) for angle in angles
    ]

    coords: List[Tuple[float, float]] = []
    lattice_vectors: List[Tuple[int, int, int, int]] = []
    for vector in itertools.product(range(-limit, limit + 1), repeat=4):
        x = y = u = v = 0.0
        for coefficient, (px, py), (qx, qy) in zip(vector, physical, perpendicular):
            x += coefficient * px
            y += coefficient * py
            u += coefficient * qx
            v += coefficient * qy
        if max(abs(u), abs(v)) <= window:
            coords.append((x, y))
            lattice_vectors.append(tuple(vector))

    if not coords:
        raise ValueError("No tiling vertices were generated; adjust --limit/--window")

    index_by_vector: Dict[Tuple[int, int, int, int], int] = {
        vector: idx for idx, vector in enumerate(lattice_vectors)
    }
    edges: List[Tuple[int, int]] = []
    for idx, vector in enumerate(lattice_vectors):
        for dim in range(4):
            neighbour = list(vector)
            neighbour[dim] += 1
            neighbour_tuple = tuple(neighbour)
            neighbour_idx = index_by_vector.get(neighbour_tuple)
            if neighbour_idx is not None:
                edges.append((idx, neighbour_idx))

    # Apply a random dihedral-8 symmetry to keep runs varied.
    # Restrict the rotation to multiples of 90° so the patch stays axis aligned.
    theta = rng.randrange(4) * math.pi / 2
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
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

    # --- Aspect ratio fix ---
    # Instead of uniform scaling, stretch to fill the rectangle
    scale_x = (WIDTH - 2 * MARGIN) / span_x
    scale_y = (HEIGHT - 2 * MARGIN) / span_y
    offset_x = MARGIN - scale_x * min_x
    offset_y = MARGIN - scale_y * min_y

    points = [(scale_x * x + offset_x, scale_y * y + offset_y) for x, y in transformed]

    return TilingPatch(points=points, edges=edges)


def nearest_neighbour_cycle(
    points: Sequence[Tuple[float, float]],
    rng: random.Random,
    bias: CycleBias | None = None,
) -> List[int]:
    """Generate an initial tour using a nearest-neighbour walk."""
    unused = list(range(len(points)))
    current = unused.pop(rng.randrange(len(unused)))
    cycle = [current]

    if bias is not None:
        bias.remaining_per_ring[bias.ring_ids[current]] -= 1

    while unused:
        if bias is None:
            cx, cy = points[current]
            next_index = min(
                unused,
                key=lambda idx: (points[idx][0] - cx) ** 2 + (points[idx][1] - cy) ** 2,
            )
        else:
            next_index = min(
                unused,
                key=lambda idx: _biased_transition_cost(current, idx, points, bias, rng),
            )
        unused.remove(next_index)
        cycle.append(next_index)
        if bias is not None:
            bias.remaining_per_ring[bias.ring_ids[next_index]] -= 1
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
        indices = list(range(n - 1))
        rng.shuffle(indices)
        for i in indices:
            a = cycle[i]
            b = cycle[i + 1]
            ax, ay = points[a]
            bx, by = points[b]
            upper = n if i > 0 else n - 1
            for j in range(i + 2, upper):
                c = cycle[j]
                d = cycle[(j + 1) % n]
                if a == d or b == c:
                    continue
                cx, cy = points[c]
                dx, dy = points[d]
                current_length = math.hypot(ax - bx, ay - by) + math.hypot(cx - dx, cy - dy)
                swapped_length = math.hypot(ax - cx, ay - cy) + math.hypot(bx - dx, by - dy)
                if swapped_length + 1e-9 < current_length:
                    cycle[i + 1 : j + 1] = reversed(cycle[i + 1 : j + 1])
                    improved = True
        if not improved:
            break


def build_cycle(
    points: Sequence[Tuple[float, float]],
    rng: random.Random,
    two_opt_rounds: int,
) -> List[int]:
    """Construct a Hamiltonian-style cycle visiting each vertex exactly once."""
    bias = compute_cycle_bias(points, rng)
    cycle = nearest_neighbour_cycle(points, rng, bias=bias)
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


def render_static(
    points: Sequence[Tuple[float, float]],
    cycle: Sequence[int],
    patch_level: int,
    cycle_level: int,
    background_level: int,
    line_width: int,
    output_path: Path,
) -> Image.Image:
    """Render the patch and cycle as a single static image."""
    image = Image.new("L", (WIDTH, HEIGHT), color=gray_level(background_level))
    draw = ImageDraw.Draw(image)

    # Draw patch vertices in faint gray
    for x, y in points:
        draw.ellipse(
            (x - 1, y - 1, x + 1, y + 1),
            fill=gray_level(patch_level),
            outline=None,
        )

    # Draw cycle in bright white
    extended_cycle = list(cycle) + [cycle[0]]
    for a, b in zip(extended_cycle[:-1], extended_cycle[1:]):
        draw.line(
            (*points[a], *points[b]),
            fill=gray_level(cycle_level),
            width=line_width,
        )

    image.save(output_path)
    return image


def render_tiling(
    points: Sequence[Tuple[float, float]],
    edges: Sequence[Tuple[int, int]],
    background_level: int,
    edge_level: int,
    vertex_level: int,
    line_width: int,
    output_path: Path,
) -> Image.Image:
    """Render only the Ammann–Beenker tiling geometry."""

    image = Image.new("L", (WIDTH, HEIGHT), color=gray_level(background_level))
    draw = ImageDraw.Draw(image)

    for a, b in edges:
        ax, ay = points[a]
        bx, by = points[b]
        draw.line((ax, ay, bx, by), fill=gray_level(edge_level), width=line_width)

    if vertex_level >= 0:
        vertex_value = gray_level(vertex_level)
        for x, y in points:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=vertex_value, outline=None)

    image.save(output_path)
    return image


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
        default=48,
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info("Screensaver tool started.")
    args = parse_args(argv)
    rng = random.Random(args.seed)
    logging.info(f"Generating Ammann–Beenker patch (limit={args.limit}, window={args.window})")
    try:
        patch = generate_ab_patch(limit=args.limit, window=args.window, rng=rng)
        points = patch.points
        logging.info(f"Generated {len(points)} vertices.")
        logging.info(f"Building Hamiltonian cycle with {args.two_opt_rounds} two-opt rounds.")
        cycle = build_cycle(points, rng=rng, two_opt_rounds=args.two_opt_rounds)
        logging.info(f"Cycle constructed with {len(cycle)} points.")
        # Choose levels: patch faint gray, cycle bright white, background nearly black
        patch_level = 3
        cycle_level = 15
        background_level = 0
        output_path = args.gif.with_suffix(".png")
        logging.info(f"Rendering static image to {output_path}")
        render_static(
            points=points,
            cycle=cycle,
            patch_level=patch_level,
            cycle_level=cycle_level,
            background_level=background_level,
            line_width=args.line_width,
            output_path=output_path,
        )
        logging.info("Static image rendering complete.")
        tiling_path = output_path.with_name(f"{output_path.stem}-tiling{output_path.suffix}")
        logging.info(f"Rendering tiling image to {tiling_path}")
        render_tiling(
            points=patch.points,
            edges=patch.edges,
            background_level=background_level,
            edge_level=min(GRAY_LEVELS - 1, patch_level + 4),
            vertex_level=patch_level,
            line_width=max(1, args.line_width - 1),
            output_path=tiling_path,
        )
        logging.info("Tiling image rendering complete.")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
    finally:
        logging.info("Screensaver tool finished.")


if __name__ == "__main__":
    main()
