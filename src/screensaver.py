"""Generate a slowly refreshing Penrose maze animation for 540×960 e-paper.

The routine below constructs a Penrose rhombus tiling using de Bruijn's
pentagrid method, stitches its vertex set into a single Hamiltonian-style
cycle via a nearest-neighbour/2-opt tour, and renders a static preview that
can be used for GIF generation on the target device. The same pentagrid is
also exported as PNG overlays to aid debugging and artistic exploration.
"""
from __future__ import annotations

import argparse
import logging
import math
import random
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


def _build_penrose_basis() -> List[Tuple[float, float]]:
    """Return the five unit vectors used for the Penrose pentagrid basis."""
    basis: List[Tuple[float, float]] = []
    for index in range(5):
        angle = 2 * math.pi * index / 5.0
        basis.append((math.cos(angle), math.sin(angle)))
    return basis


def _apply_scaling(
    points: Iterable[Tuple[float, float]],
    width: int,
    height: int,
    margin: int,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float, float, float]]:
    """Scale an iterable of points to fill the drawing rectangle with padding."""

    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    if not xs or not ys:
        raise ValueError("Penrose generator produced no vertices")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x == 0 or span_y == 0:
        raise ValueError("Degenerate Penrose patch dimensions detected")

    scale_x = (width - 2 * margin) / span_x
    scale_y = (height - 2 * margin) / span_y
    offset_x = margin - scale_x * min_x
    offset_y = margin - scale_y * min_y

    scaled = [(scale_x * x + offset_x, scale_y * y + offset_y) for x, y in points]
    return scaled, (scale_x, scale_y, offset_x, offset_y)


def _safe_mod(value: float) -> float:
    """Return a fractional offset constrained to (0, 1)."""

    frac = value % 1.0
    eps = 1e-6
    if frac < eps:
        frac += eps
    if frac > 1.0 - eps:
        frac -= eps
    return frac


def generate_penrose_patch(
    limit: int,
    offset_phase: float,
) -> Tuple[List[Tuple[float, float]], List[List[Tuple[Tuple[float, float], Tuple[float, float]]]]]:
    """Generate vertex coordinates and grid segments for a Penrose tiling patch."""

    if limit < 1:
        raise ValueError("--limit must be at least 1 for Penrose tiling generation")

    basis = _build_penrose_basis()
    gammas = [_safe_mod(offset_phase + index / 5.0) for index in range(5)]

    index_range = range(-limit, limit + 1)
    vertices: dict[Tuple[int, int, int, int, int], Tuple[float, float]] = {}
    grid_indices = [set() for _ in range(5)]

    def solve_intersection(i: int, j: int, ki: int, kj: int) -> Tuple[float, float] | None:
        ei_x, ei_y = basis[i]
        ej_x, ej_y = basis[j]
        ci = ki + gammas[i]
        cj = kj + gammas[j]
        det = ei_x * ej_y - ei_y * ej_x
        if abs(det) < 1e-12:
            return None
        x = (ci * ej_y - cj * ei_y) / det
        y = (ei_x * cj - ej_x * ci) / det
        return x, y

    for i in range(5):
        for j in range(i + 1, 5):
            for ki in index_range:
                for kj in index_range:
                    intersection = solve_intersection(i, j, ki, kj)
                    if intersection is None:
                        continue
                    x, y = intersection
                    base_indices = []
                    for idx, (bx, by) in enumerate(basis):
                        value = bx * x + by * y - gammas[idx]
                        base_indices.append(int(math.ceil(value - 1e-9)))

                    for di in (0, 1):
                        for dj in (0, 1):
                            key_list = list(base_indices)
                            key_list[i] += di
                            key_list[j] += dj
                            key = tuple(key_list)  # type: ignore[assignment]
                            px = sum(key_list[idx] * basis[idx][0] for idx in range(5))
                            py = sum(key_list[idx] * basis[idx][1] for idx in range(5))
                            vertices.setdefault(key, (px, py))

                    grid_indices[i].add(ki)
                    grid_indices[j].add(kj)

    if not vertices:
        raise ValueError("Penrose generator produced no vertices; increase --limit")

    sorted_keys = sorted(vertices.keys())
    raw_points = [vertices[key] for key in sorted_keys]
    scaled_points, transform = _apply_scaling(raw_points, WIDTH, HEIGHT, MARGIN)
    scale_x, scale_y, offset_x, offset_y = transform

    def apply_transform(point: Tuple[float, float]) -> Tuple[float, float]:
        return (point[0] * scale_x + offset_x, point[1] * scale_y + offset_y)

    bounds = (
        min(x for x, _ in raw_points) - 1.0,
        max(x for x, _ in raw_points) + 1.0,
        min(y for _, y in raw_points) - 1.0,
        max(y for _, y in raw_points) + 1.0,
    )

    def clip_line(
        normal: Tuple[float, float],
        constant: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]] | None:
        nx, ny = normal
        x_min, x_max, y_min, y_max = bounds
        intersections: List[Tuple[float, float]] = []
        if abs(ny) > 1e-12:
            for x_edge in (x_min, x_max):
                y_val = (constant - nx * x_edge) / ny
                if y_min - 1e-9 <= y_val <= y_max + 1e-9:
                    intersections.append((x_edge, y_val))
        if abs(nx) > 1e-12:
            for y_edge in (y_min, y_max):
                x_val = (constant - ny * y_edge) / nx
                if x_min - 1e-9 <= x_val <= x_max + 1e-9:
                    intersections.append((x_val, y_edge))

        unique: List[Tuple[float, float]] = []
        for candidate in intersections:
            if not any(math.hypot(candidate[0] - pt[0], candidate[1] - pt[1]) < 1e-6 for pt in unique):
                unique.append(candidate)
            if len(unique) == 2:
                break
        if len(unique) < 2:
            return None
        return apply_transform(unique[0]), apply_transform(unique[1])

    grid_segments: List[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = [[] for _ in range(5)]
    for index in range(5):
        normal = basis[index]
        for ki in sorted(grid_indices[index]):
            constant = ki + gammas[index]
            segment = clip_line(normal, constant)
            if segment is not None:
                grid_segments[index].append(segment)

    return scaled_points, grid_segments


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


def render_pentagrid_overlay(
    grid_segments: Sequence[Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]],
    output_path: Path,
) -> Image.Image:
    """Render the five Penrose grids as coloured line overlays."""

    palette = [
        (220, 70, 60),
        (60, 150, 220),
        (70, 200, 120),
        (220, 170, 60),
        (180, 90, 220),
    ]
    image = Image.new("RGB", (WIDTH, HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    for index, segments in enumerate(grid_segments):
        color = palette[index % len(palette)]
        for start, end in segments:
            draw.line((*start, *end), fill=color, width=1)
    image.save(output_path)
    return image


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Pentagrid index radius controlling the Penrose patch size (default: 5)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.6,
        help="Phase offset applied to the pentagrid shifts (default: 0.6)",
    )
    parser.add_argument(
        "--two-opt-rounds",
        type=int,
        default=16,
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
    logging.info(
        "Generating Penrose pentagrid patch (limit=%s, phase=%.3f)",
        args.limit,
        args.window,
    )
    try:
        points, grid_segments = generate_penrose_patch(
            limit=args.limit,
            offset_phase=args.window,
        )
        logging.info("Generated %d Penrose vertices.", len(points))
        logging.info(f"Building Hamiltonian cycle with {args.two_opt_rounds} two-opt rounds.")
        cycle = build_cycle(points, rng=rng, two_opt_rounds=args.two_opt_rounds)
        logging.info(f"Cycle constructed with {len(cycle)} points.")
        # Choose levels: patch faint gray, cycle bright white, background nearly black
        patch_level = 3
        cycle_level = 15
        background_level = 0
        output_path = args.gif.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
        grid_output_path = output_path.with_name(f"{output_path.stem}_pentagrid.png")
        logging.info("Rendering pentagrid overlay to %s", grid_output_path)
        render_pentagrid_overlay(grid_segments=grid_segments, output_path=grid_output_path)
        logging.info("Static image rendering complete.")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise
    finally:
        logging.info("Screensaver tool finished.")


if __name__ == "__main__":
    main()
