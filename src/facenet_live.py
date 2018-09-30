import cv2
import imutils
import argparse
import sys
from imutils.video import VideoStream, FPS
import facenet
import pickle
import time
from detect_face.face_detector import FaceDetector
import tensorflow as tf
import numpy as np


def main(args):
    
    print("[STARTING] Facenet ResNet v1 for Facial Recognition")
    print(".\n.\n.")
    print("[LOADING] Loading encondings and face detector...")
    data = pickle.loads(open(args["encondings"], "rb").read())
    face_detector = cv2.CascadeClassifier(args["cascade"])
    detector = FaceDetector(args["cascade"])
    
    print("[LOADING] Loading the Convolutional Neural Network model...")
    sess = tf.Session()
    facenet.load_model(args["model"])

    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

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
        nrof_faces = len(rects)
        if nrof_faces > 0:
            face_images = detector.extract_faces(rgb, rects)

            # Recognize the images
            feed_dict = { images_placeholder: face_images, phase_train_placeholder: False }
            embeddings_array = np.zeros((nrof_faces, embedding_size))
            embeddings_array = sess.run(embeddings, feed_dict=feed_dict)

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

    parser.add_argument("--m",
                        "-model",
                        type=str,
                        default="models/facenet/201820180402-114759/20180402-114759.pb",
                        help="Path to the CNN model")

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))