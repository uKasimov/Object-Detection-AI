from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import requests
import os
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs1 = VideoStream('http://10.10.10.29:8080/video').start()
vs2 = VideoStream('http://10.10.10.29:8080/video').start()
time.sleep(1.0)
fps = FPS().start()

n = 0
while True:
	c1 = 0
	c2 = 0
	frame1 = vs1.read()
	frame1 = imutils.resize(frame1, width=700)

	(h1, w1) = frame1.shape[:2]
	blob1 = cv2.dnn.blobFromImage(cv2.resize(frame1, (300, 300)),
		0.007843, (300, 300), 127.5)

	net.setInput(blob1)
	detections1 = net.forward()


	frame2 = vs2.read()
	frame2 = imutils.resize(frame2, width=700)

	(h2, w2) = frame2.shape[:2]
	blob2 = cv2.dnn.blobFromImage(cv2.resize(frame2, (300, 300)),
		0.007843, (300, 300), 127.5)

	net.setInput(blob2)
	detections2 = net.forward()



	for i in np.arange(0, detections1.shape[2]):
		confidence = detections1[0, 0, i, 2]

		if confidence > args["confidence"]:
			idx = int(detections1[0, 0, i, 1])
			box = detections1[0, 0, i, 3:7] * np.array([w1, h1, w1, h1])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			if format(CLASSES[idx]) == 'person':
				c1 = c1 + 1
			cv2.rectangle(frame1, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame1, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			path = 'D:/OpenServer/domains/test.loc/web'


	for i in np.arange(0, detections2.shape[2]):
		confidence = detections2[0, 0, i, 2]

		if confidence > args["confidence"]:
			idx = int(detections2[0, 0, i, 1])
			box = detections2[0, 0, i, 3:7] * np.array([w2, h2, w2, h2])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			if format(CLASSES[idx]) == 'person':
				c2 = c2 + 1
			cv2.rectangle(frame2, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame2, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			path = 'D:/OpenServer/domains/test.loc/web'



	# n = n+1
	# if n == 5:
	# 	n = 0
			requests.get('http://observer.loc/main/update-count?camera_id=1&count=' + str(c1))
			requests.get('http://observer.loc/main/update-count?camera_id=2&count=' + str(c2))

	cv2.imshow("Frame1", frame1)
	cv2.imshow("Frame2", frame2)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()