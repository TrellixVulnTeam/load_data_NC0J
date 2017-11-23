"""
Load and preprocess IMDB data.
"""
import os
import numpy as np

# Number of recorders per category, i.e., N positive reviews for training, N
# negative reviews for training, same for testing.
N = 12500


def _collect_reviews(inpath='~/data/aclImdb', outpath='~/data/imdb'):
    """
    Collect reviews into four files for convenience.

    No preprocessing is done on raw data.  The four files are train-pos.txt,
    train-neg.txt, test-pos.txt, and test-neg.txt.
    """
    inpath = os.path.expanduser(inpath)
    outpath = os.path.expanduser(outpath)

    if os.path.exists(os.path.join(outpath, 'train-pos.txt')):
        return

    for i in ['train', 'test']:
        for j in ['pos', 'neg']:
            curdir = os.path.join(inpath, i, j)
            outfile = os.path.join(outpath, '{0}-{1}.txt'.format(i, j))
            reviews = [None] * N

            print('\nReading {}'.format(curdir))
            for k, elm in enumerate(os.listdir(curdir)):
                with open(os.path.join(curdir, elm), 'r') as r:
                    s = r.read().strip()
                    reviews[k] = s

            print('\nSaving {}'.format(outfile))
            with open(outfile, 'w') as w:
                w.write('\n'.join(reviews))


def load_data(filepath='imdb/imdb.npz', rawpath='~/data/aclImdb', maxlen=400,
              embedding=None):
    filepath = os.path.expanduser(os.path.join('~/data', filepath))
    datapath = os.path.expanduser('~/data/imdb')
    rawpath = os.path.expanduser(rawpath)

    if os.path.exists(filepath):
        data = np.load(filepath)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        g = embedding
        if g is None:
            import glove
            g = glove.Glove()

        _collect_reviews()

        import nltk

        def _embedding(fpath):
            reviews = [nltk.word_tokenize(line) for line in open(fpath, 'r')]
            # maxlen-1 since we add a <bos> symbol to each sentence
            return g.embedding(reviews, maxlen=maxlen-1)

        print('\nGenerating training data')
        X_train_pos = _embedding(os.path.join(datapath, 'train-pos.txt'))
        X_train_neg = _embedding(os.path.join(datapath, 'train-neg.txt'))
        X_train = np.vstack((X_train_pos, X_train_neg))
        y_train = np.append(np.zeros(X_train_pos.shape[0]),
                            np.ones(X_train_neg.shape[0]))
        y_train = np.reshape(y_train, [-1, 1])

        print('\nGenerating testing data')
        X_test_pos = _embedding(os.path.join(datapath, 'test-pos.txt'))
        X_test_neg = _embedding(os.path.join(datapath, 'test-neg.txt'))
        X_test = np.vstack((X_test_pos, X_test_neg))
        y_test = np.append(np.zeros(X_test_pos.shape[0]),
                           np.ones(X_test_neg.shape[0]))
        y_test = np.reshape(y_test, [-1, 1])

        print('\nSaving {}'.format(filepath))
        np.savez(filepath, X_train=X_train, y_train=y_train, X_test=X_test,
                 y_test=y_test)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
