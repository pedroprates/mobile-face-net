import cv2

class FaceDetector:
    def __init__(self, extractor_path):
        self.path = extractor_path
        self.face_detector = cv2.CascadeClassifier(extractor_path)

    def detect_faces(self, frame, scaleFactor=1.3, minNeighbours=1, minSize=(30, 30)):
        rects = self.face_detector.detectMultiScale(frame, 
                                                    scaleFactor=scaleFactor, 
                                                    minNeighbours=minNeighbours,
                                                    minSize=minSize,
                                                    flags=cv2.CASCADE_SCALE_IMAGE)

        return rects

    @staticmethod
    def extract_faces(frame, rects):
        multiple_faces = []

        for x, y, w, h in rects:
            h_margin = 0.1*w
            v_margin = 0.1*h

            cropped_image = frame[y-v_margin:y+h+v_margin, x-h_margin:x+w+h_margin]
            multiple_faces.append(cropped_image)
        
        return multiple_faces
