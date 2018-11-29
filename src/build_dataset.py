import argparse
import utils
import os
import tensorflow as tf
import cv2
import numpy as np
import time
import keras
import keras.backend as K
import json

def main(args):
    print("[STARTING] Starting the code to create the dataset.")
    print(".\n.\n.")
    
    print("[LOADING] Loading the Convolutional Neural Network model...")
    type_mode = args["type"]
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

    print("[LOADING] Checking the dataset path...")
    dataset_path = args['dataset']
    dataset_path = os.path.expanduser(dataset_path)
    assert os.path.isdir(dataset_path), "Dataset folder should be the dataset root folder."
    people = [person for person in os.listdir(dataset_path) if not person.startswith('.')]

    print('[RUNNING] Building the dataset!')
    times = []
    for person in people:
        print('\t[BUILD] Building ', person)
        person_path = os.path.join(dataset_path, person)
        pics = [pic for pic in os.listdir(person_path) if (pic.endswith('jpg') or pic.endswith('jpeg'))]
        nrof_pics = len(pics)
        images = np.zeros((nrof_pics, args['image'], args['image'], 3))

        for idx, pic in enumerate(pics):
            image = cv2.imread(os.path.join(person_path, pic))
            image = cv2.resize(image, (160, 160))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[idx, :, :, :] = image_rgb / 255

        # Recognize the images
        if type_mode == 'FaceNet':
            start_time = time.time()
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            embeddings_array = sess.run(embeddings, feed_dict=feed_dict)
            times.append(time.time() - start_time)
        else:
            start_time = time.time()
            embeddings_array = model.predict(images)
            times.append(time.time() - start_time)

        output_file = os.path.join(person_path, person+'.npy')

        if (os.path.isfile(output_file)):
            os.remove(output_file)

        np.save(output_file, embeddings_array)

def parse_arguments():
    """ Parsing command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', 
                    '--dataset', 
                    type=str,
                    required=True,
                    help='Path to the dataset root folder')
    ap.add_argument('-m',
                    '--model',
                    type=str,
                    help="Path to the CNN model")

    ap.add_argument('-i',
                    '--image',
                    type=int,
                    default=160,
                    help='Size of the image')

    ap.add_argument('-t',
                    '--type',
                    type=str,
                    default="MobileFaceNet",
                    help="Which model to use to create the embeddings")
    
    ap.add_argument('-j',
                    '--json',
                    type=str,
                    default='/home/pi/Documents/TCC/face-recognition/models/mobilefacenet/model.json',
                    help='Path to the JSON containing the model structure')

    ap.add_argument('-w',
                    '--weights',
                    type=str,
                    default='/home/pi/Documents/TCC/face-recognition/models/mobilefacenet/model_weights.h5',
                    help='Path to the weights of the model')
    
    return vars(ap.parse_args())

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
