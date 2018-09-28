import cv2
import imutils
import argparse
import sys
from imutils.video import VideoStream, FPS
import pickle
import time
from detect_face.face_dectector import FaceDetector


def main(args):
    
    print("[STARTING] Facenet ResNet v1 for Facial Recognition")
    print(".\n.\n.")
    print("[LOADING] Loading encondings and face detector...")
    data = pickle.loads(open(args["encondings"], "rb").read())
    face_detector = cv2.CascadeClassifier(args["cascade"])
    detector = FaceDetector(args["cascade"])

    print("[LOADING] Starting the video stream...")
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500) # Width of the frame is confurable
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces on the frame
        rects = detector.detect_faces(gray)
        face_images = detector.extract_faces(rgb, rects)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

def parse_arguments(argv):
    """ Parsing arguments to run variables to the main
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--c",
                        "-cascade",
                        type=str,
                        default="../models/HaarCascade/haarcascade_frontalface_default.xml",
                        help="Path to the face cascade config files")
    parser.add_argument("--e",
                        "-encondings",
                        type=str,
                        default="../datasets/face-recognition-data/encodings.pickle",
                        help="Path to the serialized faces database")

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))