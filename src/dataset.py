import os
import numpy as np
import progressbar

def only_alpha(string):
    return all(not a.isdigit() for a in string)

def clean_name(path_name, with_number=False):
    path_name = path_name.split('/')[-1]
    path_name = path_name.split('.')[0]

    if with_number:
        return path_name

    list_names = path_name.split('_')

    list_names = list(filter(lambda x: only_alpha(x), list_names))
    name = '_'.join(list_names)
    return name

def get_names(data):
    """ Return the list of unique names that compose the dataset

        :params data: The dataset to be analyzed
    """
    names = []
    for image_path in (data):
        name = clean_name(image_path)

        if name not in names:
            names.append(name)
        
    return names

def build_dataset(data, 
                  output='output',
                  base_path='/Users/pedroprates/Google Drive/FaceRecognition/datasets/lfw/lfw_mtcnnpy_160'):
    people = get_names(data)
    embeddings = []
    print('[CHECK] It has %d people on the dataset.' % len(people))
    for person in progressbar.progressbar(people):
        person_path = os.path.join(base_path, person)
        person_path = os.path.join(person_path, output)

        faces = os.listdir(person_path)
        faces = [os.path.join(person_path, f) for f in faces]
        nrof_faces = len(faces)

        for idx, face in enumerate(faces):
            embedding_face = np.load(face)
            embedding = { 'name': clean_name(face, with_number=True),
                          'embedding': embedding_face }
            embeddings.append(embedding)
        
    embeddings_output_path = os.path.join(base_path, 'embeddings_test_mac.npy')
    if os.path.exists(embeddings_output_path):
        os.remove(embeddings_output_path)
        
    np.save(embeddings_output_path, np.array(embeddings))

def main():
    X_test = np.load('/Users/pedroprates/Google Drive/FaceRecognition/datasets/lfw/xtest.npy')
    print('[CHECK] Test set has %d files.' % X_test.shape[0])
    print('[STARTING] Building dataset...')
    build_dataset(X_test)

if __name__ == "__main__":
    main()