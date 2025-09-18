#!/usr/bin/env python3
"""Generate both grid and Hamiltonian path images as 540x960 PNGs.

This script generates two images:
1. grid.png - The Ammann-Beenker tiling pattern
2. path.png - The Hamiltonian path through the tiling vertices

Both images are exactly 540x960 pixels and ready for display.
"""

import sys
import random
import numpy as np
from pathlib import Path

# Import our modules
from debruijn_tiling import AmmannBeenkerTiling, generate_random_tiling
from hamiltonian_path import extract_vertices_from_tiling, build_cycle, render_hamiltonian_path


def main():
    """Generate both grid and path images."""
    print("Generating 540x960 images...")

    # Set a consistent seed for reproducible results
    seed = random.randint(1, 1000000)
    print(f"Using seed: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    # Create ONE tiling instance that will be used for both images
    print("1. Creating Ammann-Beenker tiling...")
    tiling = AmmannBeenkerTiling()
    print(f"   Grid offsets: {tiling.grid_offsets}")

    k_range = range(-15, 16)

    # Generate and save the grid image
    print("2. Generating grid image...")
    tiling.plot_tiling(
        k_range=k_range,
        figsize=(5.4, 9.6),
        line_width=1.0,
        save_path='grid.png'
    )
    print("   * Grid saved as grid.png")

    # Generate the path image using the SAME tiling instance
    print("3. Generating Hamiltonian path from same tiling...")

    # Extract vertices from the exact same tiling
    vertices = extract_vertices_from_tiling(tiling, k_range)
    print(f"   Extracted {len(vertices)} vertices")

    # Build Hamiltonian cycle
    rng = random.Random(seed)
    cycle = build_cycle(vertices, rng, two_opt_rounds=50)
    print(f"   Built cycle through {len(cycle)} vertices")

    # Render the path
    render_hamiltonian_path(
        vertices,
        cycle,
        show_vertices=False,  # Clean path without dots
        save_path='path.png'
    )

    print("   * Path saved as path.png")

    print("\nGeneration complete!")
    print("Files created:")
    print("  * grid.png - 540x960 Ammann-Beenker tiling")
    print("  * path.png - 540x960 Hamiltonian path")


if __name__ == "__main__":
    main()