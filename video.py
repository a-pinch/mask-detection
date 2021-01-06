# import the necessary packages
from threading import Thread
import sys
import cv2
import time

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue

class FileVideoStream:
	def __init__(self, path, queueSize=2):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely
		idx, ids = 0, 0
		while True:
			idx += 1
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				break

			# otherwise, ensure the queue has room in it
			if self.Q.full(): 
				self.Q.get()
				ids += 1

			# read the next frame from the file
			(grabbed, frame) = self.stream.read()
			# if the `grabbed` boolean is `False`, then we have
			# reached the end of the video file
			if not grabbed:
				print('Frame is not grabbed. Video stream is stopped.')
				self.stop()
				break
			# add the frame to the queue
			self.Q.put(frame)
			print("frames skipped  %d from %d" % (ids, idx))
		self.stream.release()

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0


	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

	def running(self):
                # return True if the thread is running
                return not self.stopped
