# content-segmentation
Repository used for the development of methods for automatic content segmentation in editorial parts

## combine.py
Used to read in text-based similarity measures and visual-based
distance/dissimilarity measures, combine them together in one second
resolution, plot the original and created dissimilarity matrices and
output predictions for segment boundaries from individual and combined
modalities.

### running

```
./combine.py --data=ina 5266518001
```

Creates files `5266518_001.png` and `5266518_001.txt`.

```
./combine.py --data=urheiluruutu 202000823730
```

Creates files `PROG_2020_00823730.png` and `PROG_2020_00823730.txt`.

### inputs

```
textual-input/<DATA>_parts_timestamps.pickle
textual-input/<DATA>_subtitles_timestamps.pickle
textual-input/<DATA>_subtitle_title_similarity.pickle
```

<DATA> is either `ina` or `urheiluruutu`.

The above files contain for all videos in the dataset the ground truth
segment boundaries, timestamps of the subtitles and a matrix of
subtitle-wise mutual similarity values.  Files are produced with
!!!ISMAIL FILL IN!!!

```
visual-input/<VIDEO_LABEL>-time.npy
visual-input/<VIDEO_LABEL>-raw.npy
visual-input/<VIDEO_LABEL>-eq.npy
```

<VIDEO_LABEL> consists of the digits of the video filenames.

The above files contain per video the end time points of the visual
shots as detected by Flow and the matrices of shot-wise mutual
distance values. The raw distance values have been provided as such
and after histogram equalization to range [0,1].  The files are
produced with PicSOM using ResNet-152 features extracted from the
middle frames of the shots.

### outputs

```
<VIDEO_FILENAME>.png
<VIDEO_FILENAME>.txt
```

The png file visualizes the input unimodal and combined bimodal
distance matrices, the segment boundary scores and the part bounsdary
ground truths.

The txt file consists of four lines like:
```
txt (48 1.0000) (13 0.9705) (267 0.8499)  ...
vis (11 1.0000) (266 0.5472) (221 0.5001) ...
sum (11 1.0000) (221 0.7681) (48 0.7172)  ...
mul (11 1.0000) (48 0.6896) (221 0.6165)  ...
```

The lines above are for textual, visual and two combined methods for
detecting the segment boundaries.  Within parentheses, the first value
is the predicted starting second for a segment and the second value is
the corresponding prediction score normalized within [0, 1].

