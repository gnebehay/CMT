import cv2
from numpy import math, hstack

import numpy as np


class FileVideoCapture(object):

	def __init__(self, path):
		self.path = path
		self.frame = 1

	def isOpened(self):
		im = cv2.imread(self.path.format(self.frame))
		return im != None

	def read(self):
		im = cv2.imread(self.path.format(self.frame))
		status = im != None
		if status:
			self.frame += 1
		return status, im

def squeeze_pts(X):
	X = X.squeeze()
	if len(X.shape) == 1:
		X = np.array([X])
	return X

def array_to_int_tuple(X):
	return (int(X[0]), int(X[1]))

def L2norm(X):
	return np.sqrt((X ** 2).sum(axis=1))

current_pos = None
tl = None
br = None

def get_rect(im, title='get_rect'):
	mouse_params = {'tl': None, 'br': None, 'current_pos': None,
		'released_once': False}

	cv2.namedWindow(title)
	cv2.moveWindow(title, 100, 100)

	def onMouse(event, x, y, flags, param):

		param['current_pos'] = (x, y)

		if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
			param['released_once'] = True

		if flags & cv2.EVENT_FLAG_LBUTTON:
			if param['tl'] is None:
				param['tl'] = param['current_pos']
			elif param['released_once']:
				param['br'] = param['current_pos']

	cv2.setMouseCallback(title, onMouse, mouse_params)
	cv2.imshow(title, im)

	while mouse_params['br'] is None:
		im_draw = np.copy(im)

		if mouse_params['tl'] is not None:
			cv2.rectangle(im_draw, mouse_params['tl'],
				mouse_params['current_pos'], (255, 0, 0))

		cv2.imshow(title, im_draw)
		_ = cv2.waitKey(10)

	cv2.destroyWindow(title)

	tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
		min(mouse_params['tl'][1], mouse_params['br'][1]))
	br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
		max(mouse_params['tl'][1], mouse_params['br'][1]))

	return (tl, br)

def in_rect(keypoints, tl, br):
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	x = keypoints[:, 0]
	y = keypoints[:, 1]

	C1 = x > tl[0]
	C2 = y > tl[1]
	C3 = x < br[0]
	C4 = y < br[1]

	result = C1 & C2 & C3 & C4

	return result

def keypoints_cv_to_np(keypoints_cv):
	keypoints = np.array([k.pt for k in keypoints_cv])
	return keypoints

def find_nearest_keypoints(keypoints, pos, number=1):
	if type(pos) is tuple:
		pos = np.array(pos)
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	pos_to_keypoints = np.sqrt(np.power(keypoints - pos, 2).sum(axis=1))
	ind = np.argsort(pos_to_keypoints)
	return ind[:number]

def draw_keypoints(keypoints, im, color=(255, 0, 0)):

	for k in keypoints:
		radius = 3  # int(k.size / 2)
		center = (int(k[0]), int(k[1]))

		# Draw circle
		cv2.circle(im, center, radius, color)

def track(im_prev, im_gray, keypoints, THR_FB=20):
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	num_keypoints = keypoints.shape[0]

	# Status of tracked keypoint - True means successfully tracked
	status = [False] * num_keypoints

	# If at least one keypoint is active
	if num_keypoints > 0:
		# Prepare data for opencv:
		# Add singleton dimension
		# Use only first and second column
		# Make sure dtype is float32
		pts = keypoints[:, None, :2].astype(np.float32)

		# Calculate forward optical flow for prev_location
		nextPts, status, _ = cv2.calcOpticalFlowPyrLK(im_prev, im_gray, pts, None)

		# Calculate backward optical flow for prev_location
		pts_back, _, _ = cv2.calcOpticalFlowPyrLK(im_gray, im_prev, nextPts, None)

		# Remove singleton dimension
		pts_back = squeeze_pts(pts_back)
		pts = squeeze_pts(pts)
		nextPts = squeeze_pts(nextPts)
		status = status.squeeze()

		# Calculate forward-backward error
		fb_err = np.sqrt(np.power(pts_back - pts, 2).sum(axis=1))

		# Set status depending on fb_err and lk error
		large_fb = fb_err > THR_FB
		status = ~large_fb & status.astype(np.bool)

		nextPts = nextPts[status, :]
		keypoints_tracked = keypoints[status, :]
		keypoints_tracked[:, :2] = nextPts

	else:
		keypoints_tracked = np.array([])
	return keypoints_tracked, status

def rotate(pt, rad):
	if(rad == 0):
		return pt

	pt_rot = np.empty(pt.shape)

	s, c = [f(rad) for f in (math.sin, math.cos)]

	pt_rot[:, 0] = c * pt[:, 0] - s * pt[:, 1]
	pt_rot[:, 1] = s * pt[:, 0] + c * pt[:, 1]

	return pt_rot

def br(bbs):

	result = hstack((bbs[:, [0]] + bbs[:, [2]] - 1, bbs[:, [1]] + bbs[:, [3]] - 1))

	return result

def bb2pts(bbs):

	pts = hstack((bbs[:, :2], br(bbs)))

	return pts
