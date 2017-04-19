# Introduction
CMT (Consensus-based Matching and Tracking of Keypoints for Object Tracking) is
a novel keypoint-based method for long-term model-free object tracking in a
combined matching-and-tracking framework.
Details can be found on the [project page](http://www.gnebehay.com/cmt)
and in our [publication](http://www.gnebehay.com/publications/wacv_2014/wacv_2014.pdf).
The Python implementation in this repository is platform-independent and runs
on Linux, Windows and OS X.

# License
CMT is freely available under the [3-clause BSD license][1],
meaning that you can basically do with the code whatever you want.
If you use our algorithm in scientific work, please cite our publication
```
@inproceedings{Nebehay2015CVPR,
    author = {Nebehay, Georg and Pflugfelder, Roman},
    booktitle = {Computer Vision and Pattern Recognition},
    month = jun,
    publisher = {IEEE},
    title = {Clustering of {Static-Adaptive} Correspondences for Deformable Object Tracking},
    year = {2015}
}
```

# Dependencies
* Python
* OpenCV-Python (>= 2.4, < 3)
* NumPy
* SciPy
* optional: ipdb (for debugging the code)

Note for Windows users: if you are unable to read video files, please follow this suggestion: http://stackoverflow.com/questions/11699298/opencv-2-4-videocapture-not-working-on-windows

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
* `--skip N` Skips N frames of the video input
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

Use a sequence of numbered image files as an input:
```
python run.py sequence_dir/{:08d}.jpg
```
Here, {:08d} is a python format string that is expanded to 00000001.jpg, 00000002.jpg, etc.

[1]: http://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_.28.22Revised_BSD_License.22.2C_.22New_BSD_License.22.2C_or_.22Modified_BSD_License.22.29
