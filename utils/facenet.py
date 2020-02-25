import os
from glob import glob
import tensorflow as tf


def get_filenames(pth):
    """
    Get the filenames of the model and checkpoint files based on the root path

    Args:
        pth: Root path where the model and ckpt files are located

    Returns:
        str: model filename
        str: checkpoint filename
    """
    os.chdir(pth)

    model_file = glob('*meta')
    assert len(model_file) > 0, "There are no model files on this root folder"

    if len(model_file) > 1:
        print('There are more than one model files on this root folder. Getting the first.')

    model_file = model_file[0]

    ckpt_file = glob('*index')
    assert len(ckpt_file) > 0, "There are no ckpt files on this root folder"

    if len(model_file) > 1:
        print('There are more than one checkpoint files on this root folder. Getting the first')

    ckpt_file = ckpt_file[0]
    ckpt_file = ckpt_file.strip('.index')

    return model_file, ckpt_file


def load_model(path, input_map = None):
    """
    Loading the tensorflow 1.x model

    Args:
        path: the path of the folder containing the model, or the model file
        input_map: input map for the graph, defaults to None

    """
    if os.path.isfile(path):
        print('Loading files:')
        print(f'Model filename: {path}')

        with tf.compat.v1.gfile.FastGFile(path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')

    else:
        model_file, ckpt_file = get_filenames(path)
        curr_sess = tf.compat.v1.get_default_session()

        print('Loading files:')
        print(f'Model: {model_file}')
        print(f'Checkpoint: {ckpt_file}')

        saver = tf.compat.v1.train.import_meta_graph(os.path.join(path, model_file),
                                                     input_map=input_map)
        saver.restore(curr_sess, os.path.join(path, ckpt_file))

    print('Succeeded!')
