# Training Data

This directory contains chess position datasets for training.

## Structure

- `raw_pgn/`: Downloaded PGN files from chess databases (if allowed by competition rules)
- `processed/`: Preprocessed datasets in NPZ or HDF5 format

## Data Format

### Processed Dataset Format (NPZ)
Each sample contains:
- `planes`: Board representation (27, 8, 8) float array
- `policy`: Move probabilities (4672,) float array  
- `value`: Game outcome (-1, 0, +1) from position's perspective
- `move_index`: Index of played move (for supervised learning)

### Position Dataset v1
Generated from PGN files using `scripts/pgn_to_dataset.py`:
- Input: PGN files with strong player games
- Output: (board_planes, move_policy, game_result) tuples
- Format: Compressed NPZ arrays for fast loading

## Usage

```bash
# Convert PGN to dataset
python ../scripts/pgn_to_dataset.py --input raw_pgn/*.pgn --output processed/dataset_v1.npz

# Verify dataset
python -c "import numpy as np; data = np.load('processed/dataset_v1.npz'); print(data.files)"
```

## Notes

- Keep PGN files organized by source/rating level
- Processed datasets should be versioned (v1, v2, etc.)
- Monitor disk space - compressed datasets can still be large
