import cv2
import imutils
import argparse
import sys
from imutils.video import VideoStream, FPS
import pickle
import time
from detect_face.face_detector import FaceDetector
import tensorflow as tf
import numpy as np
import utils


def main(args):
    
    print("[STARTING] Facenet ResNet v1 for Facial Recognition")
    print(".\n.\n.")
    print("[LOADING] Loading face detector...")
    face_detector = cv2.CascadeClassifier(args["cascade"])
    detector = FaceDetector(args["cascade"])
    
    print("[LOADING] Loading the faces dataset...")
    dataset, name_to_idx, idx_to_name = utils.build_dataset(args["dataset"])

    print("[LOADING] Loading the Convolutional Neural Network model...")
    sess = tf.Session()
    utils.load_model(args["model"])

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
        frame = imutils.resize(frame, width=500) # Width of the frame is configurable
        
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

            for idx, embedding in enumerate(embeddings_array):
                predicted = utils.predict_face(dataset, name_to_idx, idx_to_name, embedding)
                x, y, w, h = rects[idx]
                color = (255, 0, 0) if predicted == "Unknown" else (0, 255, 0)
                cv2.rectangle(frame, (x, y+h), (x+w, y), color, 2)
                top = y+h-15 if y+h-15 > 15 else y+h+15
                cv2.putText(frame, predicted, (x, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # Display the image
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approximated FPS: {:.2f}fps".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()

def parse_arguments(argv):
    """ Parsing arguments to run variables to the main
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--c",
                        "-cascade",
                        type=str,
                        default="../models/HaarCascade/haarcascade_frontalface_default.xml",
                        help="Path to the face cascade config files")
    parser.add_argument("--d",
                        "-dataset",
                        type=str,
                        default="../datasets/tcc",
                        help="Path datasets source folder")

    parser.add_argument("--m",
                        "-model",
                        type=str,
                        default="models/facenet/201820180402-114759/20180402-114759.pb",
                        help="Path to the CNN model")

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))