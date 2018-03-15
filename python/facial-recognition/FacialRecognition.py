from __future__ import print_function

from threading import Thread

import numpy as np
import cv2

import time
import uuid
import json
import os

import CaptureDevices as CD

#### USE DICT FOR COLORS ####
CLRS = {'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255)}

class Subject:
    def __init__(self, first='', last=''):
        self.first = first
        self.last = last
        self.name = ''
        self.uuid = ''
        self.id = -1
        self.folder = ''
        self.images = []
        self.details = {}

    def __repr__(self):
        return self.name

    # creates initial values for a new subject
    def init(self):
        self.uuid = str(uuid.uuid4())[-12:] # keep the last 12 of uuid
        self.name = self.last + '_' + self.first
        self.folder = self.name + '-' + self.uuid
        self.details = {'name':self.name,
                        'uuid':self.uuid,
                        'id': self.id,
                        'folder':self.folder,
                        'images': self.images,
                        }
    # sets values for an existing subjet or when updating a subject
    def update(self):
        self.name = self.last + '_' + self.first
        self.folder = self.name + '-' + self.uuid
        self.details = {'name':self.name,
                        'uuid':self.uuid,
                        'id': self.id,
                        'folder':self.folder,
                        'images': self.images,
                        }

    def get_details(self):
        if len(self.details) > 0:
            return self.details
        else:
            self.init()
            return self.details

    def set_details(self, first='', last='', folder='',
                    sid=-1, uuid='', images=[]):
        self.first = first
        self.last = last
        self.folder = folder
        self.id = sid
        self.uuid = uuid
        self.images = images
        self.update()

class FaceDetector:
    """Class used for detection of faces and eyes"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.scaleFactor = 1.5
        self.minNeighbors = 6
        self.minSize = (64, 64)
        self.faces = []
        self.results = []

    # get faces method, img should be in grayscale for optimal performance.
    def get(self, img):
        """ returns a list of dicts containing any faces and eyes found """
        self.img = img
        self.results = [] # start with empty list each time
        self.findFaces()
        self.findEyes()
        return self.results

    def findFaces(self):
        self.faces = self.face_cascade.detectMultiScale(self.img,
                                                        self.scaleFactor,
                                                        self.minNeighbors,
                                                        minSize=self.minSize,
                                                        )

    def findEyes(self):
        for face in self.faces:
            # only scan top half
            roi = self.img[face[1]:face[1]+int(face[3]*.5),
                           face[0]:face[0]+face[2]]

            # slice so we only return 2 results
            eyes = self.eye_cascade.detectMultiScale(roi)[:2]

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

    def getSubById(self, sid):
        for sub in self.subjects:
            if sub.id == sid:
                return sub

    def getSubByName(self, first, last):
        name = first + '_' + last
        for sub in self.subjects:
            if sub.name == name:
                return sub

class FileUtilities:
    def __init__(self, path='img_db', db_name='maindb.json',
                 img_size=(100,100), resize_mode=cv2.INTER_AREA):
        self.db = {}
        self.name = db_name
        self.path = path
        self.img_size = img_size
        self.resize_mode = resize_mode
        self.subjects = []
        self.images = []
        self.labels = []

    def init(self):
        self.db = self.build_db()

        self.writer(self.path, self.name, self.db, 1)
        self.db = self.load_db(self.name)
        self.subjects = self.get_subjects_from_db()
        self.load_img_and_labels()


    def get_folders(self, path):
        """ scans a path and returns a list of image folders """
        # build list of directories
        folders = [f for f in os.listdir(path)
                         if os.path.isdir(os.path.join(path, f))]
        # validate the folder name format of first_last-uuid
        image_folders = [f for f in folders if len(f.split('-')) == 2]

        # exclude folders which do not contain any .jpg files
        image_folders = [f for f in image_folders
                             if len(os.listdir(os.path.join(path,f))) > 0
                                 and '.jpg' in ''.join(
                                     os.listdir(os.path.join(path,f)))
                         ]

        return image_folders

    def build_db(self):
        """ scans files and folders in a path and returns a dictionary,
         which contains, id, name, uuid, list of any .jpg files """
        folders = self.get_folders(self.path)
        db = {}
        for folder in folders:
            idx = folders.index(folder)
            name = folder.split('-')[0] #first element
            uuid = ''.join(folder.split('-')[1:])
            files = []
            for file in os.listdir(os.path.join(self.path, folder)):
                if file.endswith('.jpg'):
                    files.append(os.path.join(self.path, folder, file))

            db[folder] = {'id':idx, 'uuid':uuid, 'name':name, 'images':files}
        return db

    def load_db(self, db_file):
        """ loads the db file in the root of the dir, the file is in json """
        db_file = os.path.join(self.path, db_file)
        with open(db_file, 'r') as db:
            data = json.load(db)
        return data

    def get_subjects_from_db(self):
        """ function used to create a list of Subject objects
         from a maindb file, returns a list of Subject objects """
        db = self.db
        subjects = []
        for folder in db.keys():
            info = db[folder]
            first = info['name'].split('_')[1]
            last = info['name'].split('_')[0]
            sid = info['id']
            uuid = info['uuid']
            images = info['images']
            sub = Subject()
            sub.set_details(first, last, folder, sid, uuid, images)
            subjects.append(sub)

        return subjects

    def load_img_and_labels(self):
        self.images = [] # clear our info
        self.labels = []
        for subject in self.subjects:
            for img in subject.images:
                try:
                    img = cv2.imread(img, 0)
                    img = cv2.resize(img, self.img_size,
                                     self.resize_mode)
                except Exception as e:
                    print(e)
                    img = np.zeros((30,30,3), np.uint8) # empty

                self.images.append(img)
                self.labels.append(subject.id)

        self.labels = np.asarray(self.labels)


    def writer(self, path, fname, data, ftype=0, mode='w'):
        """ used for writing data, can be raw, json, or cvMat(np.array) """

        current = os.path.abspath('.')
        dt = time.strftime('%a %x %H:%M:%S', time.localtime())

        try:
            os.chdir(path)
        except Exception:
            try:
                os.mkdir(path)
                os.chdir(path)
            except Exception as e:
                print(e)
                return

        try:
            if ftype == 0: # normal file
                with open(fname, mode) as f:
                    f.write(data)

            elif ftype == 1: # json file
                with open(fname, mode) as f:
                    json.dump(data, f)

            elif ftype == 2: # image file
                cv2.imwrite(fname, data)

            print(dt, 'created:', fname)

        except Exception as e:
            print(dt, "A problem has occurred while attempting to write:",
                  fname)
            print(e)

        finally:
            os.chdir(current)

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
        if TTS:
            self.tts = ttswrapper.TTSWrapper('BasicTTS.exe')
            self.tts.start()

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

    def identifySub(self, img):
        """ takes an image and returns a Subject object with
         a confidence level for the prediction """

        # resize to same size being used on our face recognizer
        img = cv2.resize(img, self.FR.img_size, self.FR.resize_mode)

        # predict the subject in the image and return confidence value
        prediction, confidence = self.FR.face_recognizer.predict(img)

        if confidence < 100: # closer to 0.0 means more confidence
            sub = self.FR.getSubById(prediction)
            return sub, confidence
        else:
            sub = Subject()
            sub.get_details()
            return sub, 0 # return empty Subject object

    def start_capture(self):
        if type(self.capture) == int:
            self.capture = CD.WebCamStream(self.capture)

        elif type(self.capture) == list or type(self.capture) == tuple:
            cam, width, height = self.capture
            self.capture = CD.WebCamStream(cam, width, height)

        elif self.capture.startswith('rtsp'):
            self.capture = CD.RtspStream(self.capture)

        elif self.capture.startswith('http'):
            host = self.capture.split('/')[2]
            url = '/'+''.join(self.capture.split('/')[3:])
            self.capture = CD.HttpStream(host, url)

        self.capture.start()
        self.next_image()

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
            subject, confidence = self.identifySub(ROI)

            # set last detection
            last_sub = img.copy()[Ly:Ry, Lx:Rx]
            self.detection_last = [subject, last_sub, time.localtime()]

            # scaling trick, 1 - (ROI width/ img width)
            # smaller for closer objects, larger for further
            font_scale = 1-float((Rx-Lx)/img.shape[1])

            # if match found
            if not subject.first == '' and not subject.last == '':

                if TTS: # if text to speech is enabled, say name.
                    self.tts.say(subject.first+', '+subject.last)

                # draw nametag
                ID = (subject.first[0] + '. ' + subject.last).title()
                cv2.putText(img, ID,(Lx, Ly-5), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, CLRS['yellow'], 2)

                # draw confidence level
                conf = str(round(confidence, 1))
                cv2.putText(img, conf, (Lx+4, Ry-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, CLRS['yellow'], 2)

                # draw face rect in green
                self.draw_bounding_rect(img, [Lx,Ly,Rx,Ry],
                                    full=False, lineSize=2, color=CLRS['green'])
            else:
                # draw nametag
                cv2.putText(img, 'unknown', (Lx, Ly-5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), 2)

                # draw face rect, draw_bounrding_rect draws in red by default
                self.draw_bounding_rect(img, [Lx,Ly,Rx,Ry], lineSize=4)

                # draw eyes
                for (eLx, eLy, eRx, eRy) in eyes:
                    # each corner needs Lx and Ly to account for the offset
                    corners = [Lx+eLx, Ly+eLy, Lx+eLx+eRx, Ly+eLy+eRy]
                    self.draw_bounding_rect(img, corners, lineSize=4, color = CLRS['yellow'])
                # save face since it is unknown
                if self.save_unknown:
                    self.save_roi_image(ROI)

            # draw recognition process time for bounding box
            if self.DEBUG:
                self.draw_processing_time(img, time_start,(Rx-50, Ry+15))

    def find_font_scale(width):
        scale = width * .01
        if scale < 1 and scale > .5:
            return scale
        elif scale > 2:
            return

    def save_roi_image(self, img, folder='unknown'):
        # image is at least 68px tall and not blurry
        try:
            if img.shape[0] < 68:
                raise Exception('Unknown not saved, image is too small')
            if blurry_image(img, thresh=self.blur_thresh):
                raise Exception('Unknown not saved, image is too blurry')
            path = os.path.join(self.db.path, folder)
            filename = str(uuid.uuid4()) + '.jpg'
            self.db.writer(path, filename, img, 2)
        except BaseException as e:
            print(self.time_stamp, e)

    def draw_bounding_rect(self, img, cords,
                         full=True, color=CLRS['red'], lineSize=2):
        Lx, Ly, Rx, Ry = cords
        if full:
            cv2.rectangle(img, (Lx,Ly), (Rx,Ry), color, lineSize)
        else:
            # top left
            cv2.line(img, (Lx,Ly) , (Lx,Ly+int(Ry*.05)), color, lineSize)
            cv2.line(img, (Lx,Ly) , (Lx+int(Rx*.05),Ly), color, lineSize)
            # top right
            cv2.line(img, (Rx,Ly) , (Rx,Ly+int(Ry*.05)), color, lineSize)
            cv2.line(img, (Rx,Ly) , (Rx-int(Rx*.05),Ly), color, lineSize)
            # bot left
            cv2.line(img, (Lx,Ry) , (Lx,Ry-int(Ry*.05)), color, lineSize)
            cv2.line(img, (Lx,Ry) , (Lx+int(Rx*.05),Ry), color, lineSize)
            # bot right
            cv2.line(img, (Rx,Ry) , (Rx,Ry-int(Ry*.05)), color, lineSize)
            cv2.line(img, (Rx,Ry) , (Rx-int(Rx*.05),Ry), color, lineSize)


    def draw_processing_time(self, img, time_start, position,
                           scale=0.5, color=(0,255,255), thickness=2):

        self.detection_time = round((cv2.getTickCount() - time_start)
                    / cv2.getTickFrequency(), 2) * 1000

        cv2.putText(img, str(self.detection_time), position,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def blurry_image(image, thresh):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    f = cv2.Laplacian(image, cv2.CV_64F).var()
    if f < float(thresh):
        print('Blurry image:', f)
        return True # blurry image
    return False
