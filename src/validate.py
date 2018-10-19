import numpy as np
import os
import math
from sklearn.model_selection import KFold


def read_pairs(path):
    pairs = []
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)

    return np.array(pairs)


def create_path(lfw_dir, pair, output):
    if len(pair) == 3:
        # TRUE
        path0 = os.path.join(lfw_dir, pair[0], output, pair[0] + '_' + '%04d' % int(pair[1])) + '.npy'
        path1 = os.path.join(lfw_dir, pair[0], output, pair[0] + '_' + '%04d' % int(pair[2])) + '.npy'
        is_same = True

    elif len(pair) == 4:
        # FALSE
        path0 = os.path.join(lfw_dir, pair[0], output, pair[0] + '_' + '%04d' % int(pair[1])) + '.npy'
        path1 = os.path.join(lfw_dir, pair[2], output, pair[2] + '_' + '%04d' % int(pair[3])) + '.npy'
        is_same = False

    else:
        raise RuntimeError('Error while reading the pair images. It was expected 3 or 4 elements per line\ '
                           'but it was found %d elements.' % len(pair))

    return path0, path1, is_same


def get_paths(lfw_dir, pairs, output):
    nrof_skipped_pairs = 0
    path_list = []
    is_same_list = []

    for pair in pairs:
        path0, path1, is_same = create_path(lfw_dir, pair, output)

        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            is_same_list.append(is_same)
        else:
            nrof_skipped_pairs += 1

    if nrof_skipped_pairs > 0:
        print("%d pairs couldn't be read." % nrof_skipped_pairs)

    return path_list, is_same_list


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

    else:
        raise RuntimeError("Distance metric not found %s. It should be either 'cosine' or 'euclidean'" % distance_metric)

    return dist


def load_embeddings(paths):
    nrof_skips = 0
    bt_size = len(paths)
    embeddings = np.zeros((bt_size, 512))

    for i, path in enumerate(paths):
        if not os.path.exists(path):
            nrof_skips += 1
            continue

        emb = np.load(path)
        embeddings[i, :] = emb

    if nrof_skips > 0:
        print("There was %d skips when trying to read the embeddings.")

    return embeddings


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if tp + fn == 0 else float(tp) / (tp + fn)
    fpr = 0 if fp + tn == 0 else float(fp) / (fp + tn)
    acc = float(tp + tn) / dist.size

    return tpr, fpr, acc


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  distance_metric='cosine',
                  subtract_mean=True,
                  nrof_folds=10):

    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]

    kfolds = KFold(n_splits=nrof_folds, shuffle=False)

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(kfolds.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]], axis=0))
        else:
            mean = 0

        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
        acc_train = np.zeros(nrof_thresholds)

        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_idx = np.argmax(acc_train)

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_idx],
                                                      dist[test_set],
                                                      actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)

    return tpr, fpr, accuracy


def evaluate(embeddings, actual_issame, distance_metric='cosine', subtract_mean=False):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, acc = calculate_roc(thresholds,
                                  embeddings1,
                                  embeddings2,
                                  np.array(actual_issame),
                                  distance_metric,
                                  subtract_mean)

    return tpr, fpr, acc


def main():
    pairs_path = '/Users/pedroprates/Google Drive/FaceRecognition/data/pairs.txt'
    lfw_path = '/Users/pedroprates/Google Drive/FaceRecognition/datasets/lfw/lfw_mtcnnpy_160'

    pairs = read_pairs(pairs_path)

    path_list, actual_issame = get_paths(lfw_path, pairs, 'output')
    embeddings = load_embeddings(path_list)
    tpr, fpr, acc = evaluate(embeddings, actual_issame, subtract_mean=True)

    print("TPR: %.2f" % tpr)
    print("FPR: %.2f" % fpr)
    print("Accuracy: %.2f" % acc)


if __name__ == '__main__':
    main()