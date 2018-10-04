import argparse
import utils
import os
import tensorflow as tf
import cv2
import numpy as np

def main(args):
    print("[STARTING] Starting the code to create the dataset.")
    print(".\n.\n.")
    print("[LOADING] Loading the model...")
    model_path = os.path.expanduser(args['model'])
    assert os.path.isfile(model_path), "The model path should be a .pb file."
    sess = tf.Session()
    utils.load_model(model_path)

    images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
    embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
    embeddings_size = embeddings.get_shape()[1]

    print("[LOADING] Checking the dataset path...")
    dataset_path = args['dataset']
    dataset_path = os.path.expanduser(dataset_path)
    assert os.path.isdir(dataset_path), "Dataset folder should be the dataset root folder."
    people = [person for person in os.listdir(dataset_path) if not person.startswith('.')]

    print('[RUNNING] Building the dataset!')
    for person in people:
        print('\t[BUILD] Building ', person)
        person_path = os.path.join(dataset_path, person)
        pics = [pic for pic in os.listdir(person_path) if (pic.endswith('jpg') or pic.endswith('jpeg'))]
        nrof_pics = len(pics)
        images = np.zeros((nrof_pics, args['image'], args['image'], 3))
        embeddings_array = np.zeros((nrof_pics, embeddings_size))

        for idx, pic in enumerate(pics):
            image = cv2.imread(os.path.join(person_path, pic))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[idx, :, :, :] = image_rgb

        feed_dict = { images_placeholder: images, phase_train_placeholder: False }
        embeddings_array = sess.run(embeddings, feed_dict=feed_dict)
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
                    required=True,
                    help="Path to the CNN model")

    ap.add_argument('-i',
                    '--image',
                    type=int,
                    default=160,
                    help='Size of the image')
    
    return vars(ap.parse_args())

if __name__ == "__main__":
    main(parse_arguments())