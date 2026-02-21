# Keyword Spotting

This project implements a keyword spotting system for the George Washington handwritten dataset using Dynamic Time Warping (DTW).

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the main script:

```
python src/main.py
```

## Pipeline

1. Preprocess pages by cropping word images using polygons.
2. Binarize and normalize word images.
3. Extract sliding-window features.
4. Compute DTW distance between feature sequences.
5. Evaluate precision, recall, and average precision.

## Output

Results and plots are stored in the `results/` folder.
