# Introduction
CMT (Consensus-based Matching and Tracking of Keypoints for Object Tracking) is
a novel keypoint-based method for long-term model-free object tracking in a
combined matching-and-tracking framework. (See [publication](http://www.gnebehay.com/publications/wacv_2014/))

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
* `--no-rotation` Disable rotation estimation
* `--bbox BBOX` Specify initial bounding box. Format: x,y,w,h
* `--pause` Specify initial bounding box.
* `--output-dir OUTPUT` Specify a directory for output data.
* `--quiet` Do not show graphical output (Useful in combination with --output-dir).

## Keys
Press any key to stop the stream. Click with the left mouse button to select the
first bounding box corner and click again the select the second.

## Examples
Define an initial bounding box:
```
python run.py --bbox=123,85,60,140 /home/cmt/test.avi
```
Use the webcam:
```
python run.py
```

