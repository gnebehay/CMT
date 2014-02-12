import cv2
import math
import numpy as np
from numpy import *

def squeeze_pts(X):
	X = X.squeeze()
	if len(X.shape) == 1:
		X = np.array([X])
	return X

def array_to_int_tuple(X):
	return (int(X[0]),int(X[1]))

def L2norm(X):
	return np.sqrt((X**2).sum(axis=1))

click_pos = None

def get_click(im, title='get_click'):
	global click_pos
	click_pos = None

	cv2.namedWindow(title)
	cv2.moveWindow(title, 100, 100)

	def onMouse(event, x, y, flags, param):
		global click_pos;

		if flags & cv2.EVENT_FLAG_LBUTTON:
			click_pos = (x,y)

	cv2.setMouseCallback(title, onMouse)

	cv2.imshow(title,im)

	#Preview
	while click_pos is None:
		key = cv2.waitKey(10)

	cv2.destroyWindow(title)

	return click_pos

current_pos = None
tl = None
br = None

def get_rect(im, title='get_rect'):

	global current_pos
	global tl
	global br
	global released_once

	current_pos = None
	tl = None
	br = None
	released_once = False

	cv2.namedWindow(title)
	cv2.moveWindow(title, 100, 100)

	def onMouse(event, x, y, flags, param):
		global current_pos
		global tl
		global br
		global released_once

		current_pos = (x,y)

		if tl is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
			released_once = True

		if flags & cv2.EVENT_FLAG_LBUTTON:
			if tl is None:
				tl = current_pos
			elif released_once:
				br = current_pos

	cv2.setMouseCallback(title, onMouse)
	cv2.imshow(title,im)

	while br is None:
		im_draw = np.copy(im)

		if tl is not None:
			cv2.rectangle(im_draw, tl, current_pos, (255,0,0))

		cv2.imshow(title, im_draw)
		key = cv2.waitKey(10)

	cv2.destroyWindow(title)

	return (tl,br)

def in_rect(keypoints, tl, br):
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	x = keypoints[:,0]
	y = keypoints[:,1]

	C1 = x > tl[0]
	C2 = y > tl[1]
	C3 = x < br[0]
	C4 = y < br[1]

	result = C1 & C2 & C3 & C4

	return result

def keypoints_cv_to_np(keypoints_cv):
	keypoints = np.array([k.pt for k in keypoints_cv])
	return keypoints

def keypoints_np_to_cv(keypoints):
	keypoints_cv = [];
	for k in keypoints:
		keypoints_cv.append((k[0], k[1]))

	return keypoints_cv

def find_nearest_keypoints(keypoints, pos, number = 1):
	if type(pos) is tuple:
		pos = np.array(pos)
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	pos_to_keypoints = np.sqrt(np.power(keypoints - pos,2).sum(axis=1))
	ind = np.argsort(pos_to_keypoints)
	return ind[:number]

#Do not use, use cv2.drawKeypoints instead
def draw_keypoints(keypoints, im, color=(255,0,0)):
	
	for k in keypoints:
		radius = 3 #int(k.size / 2)
		center = (int(k[0]), int(k[1]))

		#Draw circle
		cv2.circle(im, center, radius, color)

def track(im_prev, im_gray, keypoints, THR_FB = 20):
	if type(keypoints) is list:
		keypoints = keypoints_cv_to_np(keypoints)

	num_keypoints = keypoints.shape[0]

	#Status of tracked keypoint - True means successfully tracked
	status = [False] * num_keypoints

	#If at least one keypoint is active
	if num_keypoints > 0:
		#Prepare data for opencv:
		#Add singleton dimension
		#Use only first and second column
		#Make sure dtype is float32
		pts = keypoints[:,None,:2].astype(np.float32)

		#Calculate forward optical flow for prev_location
		nextPts,status,err = cv2.calcOpticalFlowPyrLK(im_prev, im_gray, pts)

		#Calculate backward optical flow for prev_location
		pts_back,status_back,err_back = cv2.calcOpticalFlowPyrLK(im_gray, im_prev, nextPts)

		#Remove singleton dimension
		pts_back = squeeze_pts(pts_back)
		pts = squeeze_pts(pts)
		nextPts = squeeze_pts(nextPts)
		status = status.squeeze()

		#Calculate forward-backward error
		fb_err = np.sqrt(np.power(pts_back - pts,2).sum(axis=1))

		#Set status depending on fb_err and lk error
		large_fb = fb_err > THR_FB
		status = ~large_fb & status.astype(np.bool)

		nextPts = nextPts[status,:]
		keypoints_tracked = keypoints[status,:]
		keypoints_tracked[:,:2] = nextPts

	else:
		keypoints_tracked = np.array([]) 
	return keypoints_tracked, status 

def rotate(pt, rad):
	pt_rot = np.empty(pt.shape)

	s, c = [f(rad) for f in (math.sin, math.cos)]

	pt_rot[:,0] = c*pt[:,0] - s*pt[:,1]
	pt_rot[:,1] = s*pt[:,0] + c*pt[:,1]

	return pt_rot

def overlap(T1,T2):

	#Check for equal length
	if not T1.shape[0] == T2.shape[0]:
		raise Exception('Number of entries is inconsistent')

	num_entries = T1.shape[0]

	hrzInt = minimum(T1[:, 0] + T1[:, 2], T2[:, 0] + T2[:, 2]) - maximum(T1[:, 0], T2[:, 0])
	hrzInt = maximum(0,hrzInt)
	vrtInt = minimum(T1[:, 1] + T1[:, 3], T2[:, 1] + T2[:, 3]) - maximum(T1[:, 1], T2[:, 1])
	vrtInt = maximum(0,vrtInt)
	intersection = hrzInt * vrtInt

	union = (T1[:, 2] * T1[:, 3]) + (T2[:, 2] * T2[:, 3]) - intersection

	overlap = intersection / union

	overlap[any(isnan(T1),1) | any(isnan(T2),1)] = nan

	return overlap

def br(bbs):

	result = hstack((bbs[:,[0]] + bbs[:,[2]]-1, bbs[:,[1]] + bbs[:,[3]]-1))

	return result

def tl(bbs):

	result = bbs[:,:2]

	return result

def pts2bb(pts):

	bbs = hstack((pts[:,:2], pts[:,2:4]-pts[:,:2]+1))

	return bbs

def bb2pts(bbs):

	pts = hstack((bbs[:,:2], br(bbs)))

	return pts

def write(fname, bbs):
	savetxt(fname, bbs, fmt='%.2f', delimiter=',')

def read(fname):
	bbs = genfromtxt(fname, delimiter=',')
	return bbs
