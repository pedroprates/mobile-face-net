import os
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import math
        
def load_model(model, input_map=None):
    """ Load model given its path. Currently only working with '.pb' saved models

        :param model: Path of where the model was saved
        :param input_map: Input map of the model, default to None 
    """
    model_exp = os.path.expanduser(model)
    assert os.path.isfile(model_exp), "Currently its only working with '.pb' model files. So your path should be one."

    print('Model filename: %s' % model_exp)
    with gfile.FastGFile(model_exp,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map=input_map, name='')

def distance(embeddings1, embeddings2, distance_metric='euclidean'):
    """ Calculate the distance between two embeddings. Currently working with euclidean and cosine similarity. 

        :param embeddings1: First embedding
        :param embeddings2: Second embedding
        :param distance_metric: Distance metric to be used to make the calculation. Should be either: 'euclidean' or 'cosine'

        :returns: The distance between the `embeddings1` and `embeddings2`
    """
    assert distance_metric in ['euclidean', 'cosine'], "The distance metric should be either 'euclidean' or 'cosine'"

    if distance_metric == 'euclidean':
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    elif distance_metric == 'cosine':
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi

    return dist

def build_dataset(path):
    """ Building a dataset given the path of the source folder. 
        The source folder should be structured as described on the github wiki.

        :param path: The path of the source folder

        :returns: Three dictionaries - One with the encodings, and two others mapping names to indexes.
    """ 
    dataset = {}

    people = os.listdir(path)
    for person in people:
        if person.startswith('.'):
            continue

        embs = np.load(path + '/' + person + '/' + person + '.npy')
        dataset[person] = embs

    names = iter(dataset.keys())
    idxs = iter(np.arange(len(dataset)))

    names_to_idx = dict(zip(names, idxs))
    idx_to_names = dict([x, v] for v, x in names_to_idx.items())

    return dataset, names_to_idx, idx_to_names

def get_image(dataset, name, chosen_n=-1):
    """ Given a dataset, get the chosen image in a person base. If the image index equals -1, returns a random image from the person.

        :params dataset: Dataset with known faces
        :params name: Name of the known person which the image should be returned
        :params chosen_n: The index of image from the given person. If equals to -1, returns a random image from that person.

        :returns: Embeddings from the face image of that given person
    """
    assert name in dataset.keys(), "Name not found. Make sure that your name is present on your dataset."
    
    if chosen_n == -1:
        nrof_faces = dataset[name].shape[0]
        chosen_n = np.random.randint(nrof_faces)

    chosen = dataset[name][chosen_n]
    return chosen.reshape((1, *chosen.shape))


def predict_face(dataset, name_to_idx, idx_to_name, face, threshold=.1, distance_metric='euclidean'):
    """ Given the embeddings of a face and the dataset of known embeddings, predict if the person is present on our dataset or not.

        :params dataset: Dataset with the known faces and their names
        :params name_to_idx: Dictionary with the mapping name to idx
        :params idx_to =_name: Dictionary with the mapping idx to name
        :params face: Array with the embeddings of a face
        :params threshold: Minimum acceptable distance between a known face and the face, if there aren't any known
                        faces that fulfill this requirement, it will be predicted as "Unknown"
        :params distance_metric: Distance metric to be used to make the calculation. Should be either 'euclidean' or 'cosine'

        :returns: Name of the person, if present on the dataset, or "Unknown" if it does not meet the requirements
    """
    distances = np.zeros(len(dataset))

    for person in dataset.keys():
        nrof_images = len(dataset[person])
        for image in range(nrof_images):
            known_face = get_image(dataset, person, image)
            d = distance(face, known_face, distance_metric=distance_metric)

            if distances[name_to_idx[person]] == 0:
                distances[name_to_idx[person]] = d
            elif distances[name_to_idx[person]] > d:
                distances[name_to_idx[person]] = d

    idx_min = distances.argmin()
    if distances[idx_min] > threshold:
        return 'Unknown'
    print(idx_to_name)
    print(distances)

    return idx_to_name[idx_min].replace('_', ' ')