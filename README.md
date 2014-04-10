# Introduction
CMT (Consensus-based Matching and Tracking of Keypoints for Object Tracking) is
a novel keypoint-based method for long-term model-free object tracking in a
combined matching-and-tracking framework.
Details can be found on the [project page](http://www.gnebehay.com/cmt)
and in our [publication](http://www.gnebehay.com/publications/wacv_2014/wacv_2014.pdf).

If you use our algorithm in scientific work, please cite our publication
```
@inproceedings{Nebehay2014WACV,
    author = {Nebehay, Georg and Pflugfelder, Roman},
    booktitle = {Winter Conference on Applications of Computer Vision},
    month = mar,
    publisher = {IEEE},
    title = {Consensus-based Matching and Tracking of Keypoints for Object Tracking},
    year = {2014}
}
```

# Dependencies
* Python
* OpenCV-Python (>= 2.4)
* NumPy
* SciPy

# Usage
```
usage: run.py [-h] [--challenge] [--preview] [--no-preview] [--no-scale]
               [--no-rotation] [--bbox BBOX] [--pause] [--output-dir OUTPUT]
               [--quiet]
               [inputpath]
```

## Optional arguments
* `inputpath` The input path.
* `-h, --help` show help message and exit
* `--challenge` Enter challenge mode.
* `--preview` Force preview
* `--no-preview` Disable preview
* `--no-scale` Disable scale estimation
* `--with-rotation` Enable rotation estimation
* `--bbox BBOX` Specify initial bounding box. Format: x,y,w,h
* `--pause` Pause after each frame
* `--output-dir OUTPUT` Specify a directory for output data.
* `--quiet` Do not show graphical output (Useful in combination with --output-dir).

## Object Selection
Press any key to stop the preview stream. Left click to select the
top left bounding box corner and left click again to select the bottom right corner.

## Examples
When using a webcam, no arguments are necessary:
```
python run.py
```

When using a video, the path to the file has to be given as an input parameter:
```
python run.py /home/cmt/test.avi
```

It is also possible to specify the initial bounding box on the command line.
```
python run.py --bbox=123,85,60,140 /home/cmt/test.avi
```

