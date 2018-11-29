import numpy as np
from scipy import misc
from keras.utils import Sequence


class TCCGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size)).astype(int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx+1) * self.batch_size]

        embeddings = np.array([np.load(filename) for filename in batch_y])
        images = np.array([misc.imread(filename) for filename in batch_x])
        images = images / 255

        return images, embeddings.reshape(embeddings.shape[0], -1)
