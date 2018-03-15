import io
import os
import time
import httplib
import cv2
import numpy as np

from threading import Thread
from Queue import Queue
from PIL import Image as PI

# Empty numpy array used as a placeholder for an image
BASE_IMG = np.zeros([10, 10, 3], dtype=np.uint8)

class HttpStream(Thread):
    """ Threaded class for retrieving images via a GET request """

    def __init__(self, host, url, max_q_size=256):
        self.host = host
        self.url = url
        self.Q = Queue(maxsize=max_q_size)
        self.cap_sleep = 0  # Paced capture rate
        self.stats = []
        self.captured = False
        self.stopped = False
        self.conn = httplib.HTTPConnection(host)

    def run(self):
        # Keep capturing images until the thread is terminated
        while True:

            if self.stopped:
                return

            t0 = time.time()
            self.captured, img = self.capture()


            if self.captured:
                self.Q.put(img)

            self.stats.append(time.time() - t0)

            if len(self.stats) > 30:
                self.stats.pop(0)

            time.sleep(self.cap_sleep)

    def capture(self):
        self.conn.request('GET', self.url)
        resp = self.conn.getresponse()
        if resp.status == 200:

            # Wrap in IoBytes object for PIL
            img_bytes = io.BytesIO(resp.read())
            pil_img = PI.open(img_bytes)

            # Convert to an OpenCV matrix
            try:
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                return False, BASE_IMG

            return True, img

        print(r.status)
        raise ConnectionError
        return False, BASE_IMG

    def read(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True
        self.conn.close()
        del self.Q


class LocalCapture(Thread):
    """ Threaded class for capturing images via a local capture device. """

    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = BASE_IMG
        self.ret = False
        self.stopped = False
        self.stats = []

    def run(self):
        # Keep capturing images until the thread is stopped
        while True:

            if self.stopped:
                return

            t0 = time.time()
            self.ret, self.frame = self.stream.read()

            self.stats.append(time.time() - t0)

            if len(self.stats) > 30:
                self.stats.pop(0)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class RTSPDeviceStream(Thread):
    """ Threaded class for capturing images via an RTSP stream """

    def __init__(self, url):
        self.url = url
        self.stream = cv2.VideoCapture(self.url)
        self.frame = BASE_IMG
        self.ret = False
        self.stopped = False
        self.stats = []

    def run(self):
        # Keep grabbing frames until the thread is stopped
        while True:

            if self.stopped:
                return

            t0 = time.time()
            self.ret, self.frame = self.stream.read()

            self.stats.append(time.time() - t0)

            if not self.ret: # dont give up
                self.stream.open(self.url)

            if len(self.stats) > 30:
                self.stats.pop(0)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

