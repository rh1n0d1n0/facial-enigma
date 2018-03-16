import io
import os
import time
import cv2
import numpy as np

from http.client import HTTPConnection
from threading import Thread
from queue import Queue
from PIL import Image

# Empty numpy array used as a placeholder for an image
BASE_IMG = np.zeros([100, 100, 3], dtype=np.uint8)

class Capture(Thread):
    """Base class for a capture object"""

    def __init__(self, max_q_size=255):
        super().__init__()
        self.Q = Queue(maxsize=max_q_size)
        self.capture_sleep = 0
        self.stats = {}
        self.captured = False
        self.stopped = False
        self.empty = BASE_IMG
        self.id = self.gen_id()

    def run(self):
        while True:

            if self.stopped:
                return

            self.captured, image = self.capture()

            if self.captured:
                self.Q.put(image)

            time.sleep(self.capture_sleep)

    def capture(self):
        return False, self.empty

    def read(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def gen_id(self):
        """Returns an id consisting of 16 random ASCII characters"""
        return ''.join([chr(random.randint(33, 126)) for n in range(16)])


class HttpStream(Capture):
    """Class for retrieving images via a GET request"""

    def __init__(self, host, url):
        super().__init__()
        self.host = host
        self.url = url
        self.conn = HTTPConnection(host)

    def capture(self):
        self.conn.request('GET', self.url)
        resp = self.conn.getresponse()

        if resp.status == 200:

            # Wrap image in IoBytes object for PIL
            image = io.BytesIO(resp.read())
            image = Image.open(image)

            # Convert image to OpenCV matrix
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            return True, image

        return False, self.empty

    def stop(self):
        self.stopped = True
        self.conn.close()


class LocalCapture(Capture):
    """Class for capturing images via a local capture device"""

    def __init__(self, device=0, width=640, height=480):
        super().__init__()
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def capture(self):
        return self.cap.read()

    def stop(self):
        self.stopped = True
        self.cap.release()


class RtspStream(Capture):
    """Threaded class for capturing images via an RTSP stream"""

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.cap = cv2.VideoCapture(self.url)

    def capture(self):
        return self.cap.read()

    def stop(self):
        self.stopped = True
        self.cap.release()
