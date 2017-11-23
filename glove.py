"""
Preprocess GloVe embedding data.
"""
import os
import json
import numpy as np

from sklearn.externals import joblib
from multiprocessing import current_process


class Glove:
    """
    Wrapper for GloVe embedding.
    """
    num_words = 2196018
    embedding_dim = 300

    def __init__(self, glove='~/data/glove', rawfile='glove.840B.300d.txt',
                 rebuild=False):
        """Init GloVe wrapper.

        :param glove: directory for glove data.
        :param rawfile: raw glove embedding file.
        """
        row = self.num_words
        dim = self.embedding_dim

        glove = os.path.expanduser(glove)
        rawfile = os.path.join(glove, rawfile)

        if rebuild:
            print('\nReading {}'.format(rawfile))

            with open(rawfile, 'r') as f:
                id2word = [''] * (row + 2)
                word2id = {}
                id2vec = np.empty((row + 2, dim), dtype=np.float32)
                for i, line in enumerate(f):
                    print('{0:8d}/{1}'.format(i+1, row), end='\r')
                    kv = line.split(' ', maxsplit=dim)
                    k = kv[0]
                    v = np.array(kv[1:]).astype(np.float32)
                    id2word[i] = k
                    word2id[k] = i
                    id2vec[i] = v

                id2word[-2] = '<bos>'
                word2id['<bos>'] = row
                id2vec[row] = np.ones(dim)

                id2word[-1] = '<pad>'
                word2id['<pad>'] = row + 1
                id2vec[row + 1] = np.zeros(dim)

            print('\nSaving id2word')
            with open(os.path.join(glove, 'id2word.txt'), 'w') as f:
                f.write('\n'.join(id2word))

            print('\nSaving word2id')
            with open(os.path.join(glove, 'word2id.txt'), 'w') as f:
                f.write(json.dumps(word2id))

            print('\nSaving id2vec')
            np.save(os.path.join(glove, 'id2vec.npy'), id2vec)
        else:
            print('\nLoading id2word')
            with open(os.path.join(glove, 'id2word.txt'), 'r') as f:
                id2word = [line.strip() for line in f]

            print('\nLoading word2id')
            with open(os.path.join(glove, 'word2id.txt'), 'r') as f:
                word2id = json.loads(f.read())

            print('\nLoading id2vec')
            id2vec = np.load(os.path.join(glove, 'id2vec.npy'))

        self._id2word = np.array(id2word, dtype=str)
        self._word2id = word2id
        self._id2vec = id2vec

    def embedding(self, texts, maxlen=0):
        """Get embedded representation of tokens.

        :param texts: 2D array.  Each row is a list of words, and they need
            not to have the same length.
        :param maxlen: Sentences that are longer than maxlen will be
            truncated, shorter ones will be padded.  If 0, maxlen is set to
            the max length of all the input sentences.

        :returns: 3D array, [N, maxlen, D], D is the embedding dimension.
        """
        if 0 == maxlen:
            maxlen = len(max(texts, key=len))

        word2id, id2vec = self._word2id, self._id2vec
        dim = id2vec.shape[1]
        texts = np.array(texts, dtype=str)

        print('\nAllocating embedding')
        vec = np.tile(id2vec[word2id['<pad>']], (len(texts) * (maxlen+1), 1))
        vec = np.reshape(vec, (len(texts), maxlen+1, dim))
        vec = vec.astype(np.float32)

        print('\nDo embedding...')
        for i, text in enumerate(texts):
            text = np.append(['<bos>'], text)
            for j, word in enumerate(text[:(maxlen+1)]):
                if word not in word2id:
                    word = '<unk>'
                vec[i, j] = id2vec[word2id[word]]
        return vec

    def reverse_embedding(self, vecs, k=3, embedding=True, n_process=-1):
        """Lookup nearest tokens given an embeddings.

        :param vecs: 3D array, [N, maxlen, D], N is the number of sentences,
            maxlen is maxlen for each sentences, D is the embedding dimension
            (fixed to 300).
        :param k: int, number of nearest tokens to return.
        :param embedding: bool, return embeddings as well if True.
        :param n_process: int, number of processes to spawn at the same time.
            If negative, spawn at most 8 processes.

        :returns: 4D array, [N, maxlen, k, D].
        """
        import multiprocessing
        from multiprocessing import Process

        vecs = np.array(vecs, dtype=str)
        if n_process < 0:
            n_process = min(multiprocessing.cpu_count(), 8)

        modelpath = os.path.expanduser('~/data/glove/glove-knn.pkl')

        if not os.path.exists(modelpath):
            from sklearn.neighbors import NearestNeighbors

            print('\nTraining knn')
            knn = NearestNeighbors(n_neighbors=k, p=1, n_jobs=-1)
            knn.fit(self._id2vec)

            print('\nSaving GloVe knn')
            joblib.dump(knn, modelpath)

        print('\nSearch for the words')

        manager = multiprocessing.Manager()
        retval = manager.dict()

        B, L, D = vecs.shape
        assert(D == self.embedding_dim)

        step = int(np.ceil(B*L / n_process))
        n_process = min(n_process, int(np.ceil(B*L / step)))

        print('\nSpawning {} processes'.format(n_process))

        vecs = np.reshape(vecs, [-1, D])
        procs = [Process(target=_worker,
                         args=(vecs[i*step : min((i+1)*step, B*L)]
                               .reshape(-1, D), k, modelpath, i, retval))
                 for i in range(n_process)]

        for p in procs:
            p.start()
        for p in procs:
            p.join()

        print("\nDone searching")

        res = [np.concatenate(retval[i], axis=0) for i in range(n_process)]
        inds = np.concatenate(res, axis=0)
        words = np.reshape(self._id2word[inds], [-1, L, k])

        if embedding:
            vecs = np.reshape(self._id2vec[inds], [-1, L, k, D])
            return words, vecs

        return words


def _worker(vecs, k, modelpath, curid, retval):
    """Worker to search for nearest embeddings.
    """
    cur = current_process()
    print('\n{} loading knn'.format(cur.name))
    knn = joblib.load(modelpath)
    print('\n{} searching...'.format(cur.name))
    res = knn.kneighbors(vecs, n_neighbors=k, return_distance=False)
    retval[curid] = res


if __name__ == '__main__':
    glove = Glove()
    eb = glove.embedding(['hello world !'.split()])
    print('\nembedding for "hello world"\n\n{}'.format(eb))
    print('\nshape: {}'.format(eb.shape))

    words, vecs = glove.reverse_embedding(eb, k=3)
    print(words)
