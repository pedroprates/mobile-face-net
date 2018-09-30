import cv2
import numpy as np

class FaceDetector:
    def __init__(self, extractor_path):
        self.path = extractor_path
        self.face_detector = cv2.CascadeClassifier(extractor_path)

    def detect_faces(self, frame, scaleFactor=1.3, minNeighbors=1, minSize=(30, 30)):
        rects = self.face_detector.detectMultiScale(frame, 
                                                    scaleFactor=scaleFactor, 
                                                    minNeighbors=minNeighbors,
                                                    minSize=minSize,
                                                    flags=cv2.CASCADE_SCALE_IMAGE)

        return rects

    @staticmethod
    def extract_faces(frame, rects, size=(160, 160)):
        nrof_images = len(rects)
        images = np.zeros((nrof_images, *size, 3), dtype=np.uint8)

        for idx, (x, y, w, h) in enumerate(rects):
            h_margin = int(0.1*w)
            v_margin = int(0.1*h)

            cropped_image = frame[y-v_margin:y+h+v_margin, x-h_margin:x+w+h_margin]
            images[idx,:,:,:] = cv2.resize(cropped_image, size)
        
        return images
