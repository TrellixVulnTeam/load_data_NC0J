import os
import sys
import pickle
import tarfile
from urllib.request import urlopen

import numpy as np


def load_data(fpath='/var/tmp/cifar10.npz', rescale=True, channels_last=True,
              onehot=True):
    if not os.path.exists(fpath):
        _mkdata(fpath)

    db = np.load(fpath)
    X_train, y_train = db['X_train'], db['y_train']
    X_test, y_test = db['X_test'], db['y_test']

    if rescale:
        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.

    if channels_last:
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    def _to_categorical(x, n_classes):
        x = np.array(x, dtype=int).ravel()
        n = x.shape[0]
        ret = np.zeros((n, n_classes))
        ret[np.arange(n), x] = 1
        return ret

    if onehot:
        y_train = _to_categorical(y_train, 10)
        y_test = _to_categorical(y_test, 10)

    return (X_train, y_train), (X_test, y_test)


def _mkdata(fpath):
    tmpdir = '/var/tmp'

    tmp = os.path.join(tmpdir, 'cifar10.tar.gz')
    _download(tmp, 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')

    with tarfile.open(tmp) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, tmpdir)

    tmpdir = os.path.join(tmpdir, 'cifar-10-batches-py')

    n_train = 50000
    X_train = np.empty((n_train, 3, 32, 32), dtype='uint8')
    y_train = np.empty((n_train,), dtype='uint8')

    def _load_batch(fpath, label_key='labels'):
        f = open(fpath, 'rb')
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
        f.close()
        data = d['data']
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    for i in range(1, 6):
        print('\nLoading batches {0}/{1}'.format(i, 6))
        tmp = os.path.join(tmpdir, 'data_batch_{}'.format(i))
        (X_train[(i-1)*10000 : i*10000, :, :, :],
         y_train[(i-1)*10000 : i*10000]) = _load_batch(tmp)

    tmp = os.path.join(tmpdir, 'test_batch')
    X_test, y_test = _load_batch(tmp)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    np.savez_compressed(fpath, X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)


def _download(fpath, url):
    if os.path.exists(fpath):
        return

    print('Downloading cifar10 dataset')
    workdir = os.path.dirname(fpath) or '.'
    os.makedirs(workdir, exist_ok=True)
    with urlopen(url) as ret, open(fpath, 'wb') as w:
        w.write(ret.read())


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    print('X_train shape: {0}'.format(X_train.shape))
    print('y_train shape: {0}'.format(y_train.shape))
    print('X_test shape: {0}'.format(X_test.shape))
    print('y_test shape: {0}'.format(y_test.shape))
