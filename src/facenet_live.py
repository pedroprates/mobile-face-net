import cv2
import imutils
import argparse
from imutils.video import VideoStream, FPS
import time
from detect_face.face_detector import FaceDetector
import tensorflow as tf
import utils
import keras
import keras.backend as K
import json

def main(args):
    
    print("[STARTING] Facenet ResNet v1 for Facial Recognition")
    print(".\n.\n.")
    print("[LOADING] Loading face detector...")
    detector = FaceDetector(args["cascade"])
    
    print("[LOADING] Loading the faces dataset...")
    dataset, name_to_idx, idx_to_name = utils.build_dataset(args["dataset"])

    print("[LOADING] Loading the Convolutional Neural Network model...")
    type_mode = args["type"]
    use_pi = args['run'] == 'raspberry'
    assert type_mode in ["MobileFaceNet", "FaceNet"], "Only MobileFaceNet or FaceNet are supported."

    if type_mode == 'FaceNet':
        start = time.time()
        sess = tf.Session()
        utils.load_model(args["model"])

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        print("[LOADING] Loading the FaceNet weights took %.2f" % (time.time() - start))
    else:
        K.clear_session()
        define_keras_functions()
        with open(args["json"]) as f:
            start = time.time()
            model_json = json.load(f)
            model = keras.models.model_from_json(model_json)
            print("[LOADING] Loadng the Weights...")
            model.load_weights(args["weights"])
            print("[LOADING] Loading the MobileFaceNet weights took %.2fs" % (time.time() - start))

    print("[LOADING] Starting the video stream...")
    if use_pi:
        vs = VideoStream(usePiCamera=True).start()
    else:
        vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    times = []

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
            face_images = face_images / 255
            # Recognize the images
            if type_mode == 'FaceNet':
                start_time = time.time()
                feed_dict = {images_placeholder: face_images, phase_train_placeholder: False}
                embeddings_array = sess.run(embeddings, feed_dict=feed_dict)
                times.append(time.time() - start_time)
            else:
                start_time = time.time()
                embeddings_array = model.predict(face_images)
                times.append(time.time() - start_time)

            for idx, embedding in enumerate(embeddings_array):
                embedding = embedding.reshape((1, *embedding.shape))
                predicted = utils.predict_face(dataset,
                                               name_to_idx,
                                               idx_to_name,
                                               embedding,
                                               threshold=3,
                                               distance_metric='cosine')
                x, y, w, h = rects[idx]
                color = (0, 0, 255) if predicted == "Unknown" else (0, 255, 0)
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
    print("[INFO] approximated forward propagation time: {:.2f}s".format(sum(times)/len(times)))

    cv2.destroyAllWindows()
    vs.stop()


def parse_arguments():
    """ Parsing arguments to run variables to the main
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c",
                        "--cascade",
                        type=str,
                        default="/home/pi/Documents/TCC/face-recognition/models/haarcascade/haarcascade_frontalface_default.xml",
                        help="Path to the face cascade config files")
    parser.add_argument("-d",
                        "--dataset",
                        type=str,
                        default="../datasets/tcc",
                        help="Path datasets source folder")

    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default="/home/pi/Documents/TCC/face-recognition/models/facenet/20180402-114759.pb",
                        help="Path to the CNN model")

    parser.add_argument("-t",
                        "--type",
                        type=str,
                        default="MobileFaceNet",
                        help="CNN architecture to be used")

    parser.add_argument("-j",
                        "--json",
                        type=str,
                        default="/home/pi/Documents/TCC/face-recognition/models/mobilefacenet/model.json",
                        help="Path to the JSON file")
    

    parser.add_argument("-w",
                        "--weights",
                        type=str,
                        default="/home/pi/Documents/TCC/face-recognition/models/mobilefacenet/model_weights.h5",
                        help="Path to the weights")

    parser.add_argument("-r",
                        "--run",
                        type=str,
                        default="raspberry",
                        help="Where to run, either Raspberry or PC")
    return vars(parser.parse_args())


def define_keras_functions():
    def distillation_loss(y_true, y_pred):
        return K.square(y_pred - y_true)

    def max_diff(y_true, y_pred):
        return K.max(K.square(y_pred - y_true), axis=-1)

    def sum_diff(y_true, y_pred):
        return K.sum(K.square(y_pred - y_true), axis=-1)

    keras.losses.distillation_loss = distillation_loss
    keras.metrics.max_diff = max_diff
    keras.metrics.sum_diff = sum_diff


if __name__ == "__main__":
    main(parse_arguments())
