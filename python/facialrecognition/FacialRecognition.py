import time
import uuid
import json
import os
import cv2

import numpy as np

from threading import Thread

class Subject(Object):
    """Class representation of a subject"""

    def __init__(self):
        self.name = ''
        self.details = {}

    def __repr__(self):
        return self.name

    def update(self):
        self.name = '{0} {1}'.format(self.first, self.last)
        self.details = {'first': self.first, 'last': self.last,
                        'name': self.name, 'uuid': self.uuid}

    def get(self):
        return self.details

    def set(self, first='', last='', uuid=''):
        self.first = first
        self.last = last
        self.uuid = uuid if uuid else str(uuid.uuid4())
        self.update()

class FaceDetector:
    """Face and eye detection class"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.face_detect = self.face_cascade.detectMultiScale
        self.eye_detect = self.eye_cascade.detectMultiScale
        self.scaleFactor = 1.5
        self.minNeighbors = 6
        self.minSize = (64, 64)
        self.faces = []
        self.results = []

    # Get faces method, img should be in grayscale for optimal performance.
    def get(self, img):
        """Returns dicts containing any faces and eyes found
           args:
           img - an OpenCV matrix"""

        self.img = img
        self.results = [] # Start with empty list each time
        self.find_faces()
        self.find_eyes()
        return self.results

    def find_faces(self):
        self.faces = self.face_detect(self.img, self.scaleFactor,
                                      self.minNeighbors, minSize=self.minSize)

    def find_eyes(self):
        for face in self.faces:
            # Only scan the top half of the face
            roi = self.img[face[1]:face[1] + int(face[3]*.5), face[0]:face[0] + face[2]]

            # Only return the first 2 results
            eyes = self.eye_detect(roi)[:2]

            self.results.append({'face':face, 'eyes':eyes})

class FaceRecognizer:
    def __init__(self, subjects=[], images=[], labels=[]):
        self.face_recognizer = cv2.face.createLBPHFaceRecognizer()
        self.img_size = (100, 100)
        self.resize_mode = cv2.INTER_LINEAR
        self.subjects = subjects
        self.images = images
        self.labels = labels
        self.ready = False

    def init(self):
        self.train()
        self.ready = True

    def train(self):
        self.face_recognizer.train(self.images, self.labels)

    def get_by_uuid(self, uuid):
        for sub in self.subjects:
            if sub.uuid == uuid:
                return sub

    def get_by_name(self, first, last):
        name = '{0} {1}'.format(first, last)
        for sub in self.subjects:
            if sub.name == name:
                return sub


class FacialRecognition:

    def __init__(self, cap_dev, db_path='img_db', db_name='maindb.json'):
        self.capture = cap_dev
        self.db_name = db_name
        self.db_path = db_path
        self.db = FileUtilities(self.db_path, self.db_name)
        self.detector = FaceDetector()
        self.FR = FaceRecognizer()
        self.image = np.zeros((400,400,3), np.uint8)
        self.blur_thresh = 50
        self.crop_region = [0, 0, 0, 0]
        self.image_crop = False
        self.save_unknown = True
        self.capture_time = cv2.getTickCount()
        self.detection_time = 0.01
        self.detection_last = []
        self.time_stamp_str = '%a %x %H:%M:%S'
        self.time_stamp = time.strftime(self.time_stamp_str, time.localtime())
        self.DEBUG = True

    def init(self):
        self.start_capture()
        self.db.init()
        self.FR.subjects = self.db.subjects
        self.FR.images = self.db.images
        self.FR.labels = self.db.labels
        self.FR.init()

    def update(self):
        self.detect_face()

    def next_image(self):
        self.capture_time = cv2.getTickCount()
        self.image = self.capture.read()

        if self.image_crop:
            self.crop_image()

        self.capture_time = round(((cv2.getTickCount() - self.capture_time)
                             /cv2.getTickFrequency()), 2) * 1000


    def crop_image(self):
        """ image cropping function """
        Lx, Ly, Rx, Ry = self.crop_region

        # only crop if possible (cropped size > 0)
        if Rx > Lx and Ry > Ly:
            self.image =  self.image[Ly:Ry, Lx:Rx]

    def identify(self, img):
        """Takes an image and returns a subject and confidence level
           args:
           img - an OpenCV matrix"""

        # Resize to same size being used on our face recognizer
        img = cv2.resize(img, self.FR.img_size, self.FR.resize_mode)

        # Predict the subject in the image and return confidence value
        prediction, confidence = self.FR.face_recognizer.predict(img)

        # A value closer to 0 indicates higher confidence
        if confidence < 100:
            sub = self.FR.get_by_uuid(prediction)
            return sub, confidence

        # Return empty object if confidence is low
        sub = Subject.set()
        return sub, 1000

    def detect_face(self):
        time_start = cv2.getTickCount()
        img = self.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = self.detector.get(gray)

        # do nothing if faces aren't detected 
        if len(results) == 0:
            return

        for r in results:
            face = r['face']
            eyes = r['eyes']

            # bounding rect for face and ROI
            Lx, Ly, Rx, Ry = face[0], face[1], face[0]+face[2], face[1]+face[3]
            ROI = gray[Ly:Ry, Lx:Rx]

            # try to match the face to a known subject
            subject, confidence = self.identify(ROI)


def blurry_image(image, thresh):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    f = cv2.Laplacian(image, cv2.CV_64F).var()
    if f < float(thresh):
        print('Blurry image:', f)
        return True # blurry image
    return False
