# Quasicrystal Screensaver

This repository contains a Python utility that generates a slowly unfolding,
Hamiltonian-style maze on top of an Ammann–Beenker tiling patch sized for the
540×960 M5 PaperS3 e-paper panel. Each run synthesises a fresh loop by cutting
a patch of the tiling via the classic 4D cut-and-project method, connecting all
vertices with a nearest-neighbour/2-opt tour, and rendering the segments in
discrete 10 second bursts. Output is recorded as a 16-level grayscale GIF so it
can be previewed on a workstation or pumped directly to the device frame
buffer.

## Quick start

```bash
pip install -r <(pipenv lock -r)
python src/screensaver.py --dry-run --gif preview.gif
```

On the target M5 PaperS3 drop the `--dry-run` flag and optionally provide a
`--frames-dir` so the intermediate BMP frames can be flashed to the display
hardware between the ~10 second pauses.

## Parameters

* `--limit` – cut-and-project search radius (default: 7).
* `--window` – acceptance window radius in perpendicular space (default: 1.6).
* `--two-opt-rounds` – capped number of 2-opt refinement passes over the tour.
* `--max-refreshes` – upper bound on the number of refresh frames.
* `--segments-per-refresh` – override the chunk size used per partial refresh.
* `--sleep` – seconds between refreshes (default: 10).
* `--line-width` – stroke thickness in pixels (default: 2).
* `--ink-level` / `--background-level` – 4-bit grayscale levels for ink/background.
* `--gif` – output GIF file (default: `maze_animation.gif`).
* `--frames-dir` – optional directory for BMP frames destined for the device.
* `--seed` – control randomness for repeatable mazes.
* `--dry-run` – skip the real-time sleeps so previews render immediately.
