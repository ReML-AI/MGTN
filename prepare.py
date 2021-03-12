import pickle
import numpy as np
import scipy as sp
from scipy.sparse import linalg
import argparse


def load_adj(adj_file='data/coco/coco_adj.pkl'):
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    return (_adj, _nums)


def eigenvector_centrality(adj):
    import networkx as nx
    graph = nx.from_numpy_matrix(adj)
    centrality = nx.eigenvector_centrality(graph)
    return np.array(tuple(centrality.values()))


def rowmul(arr2d, arr1d):
    return np.array(arr2d) * np.array(arr1d)[:, None]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def eigs(adj):
    eigenvalue, eigenvector = linalg.eigs(adj, k=1, which='LR')
    return (eigenvalue, eigenvector)


def adjust(adj, t=0.4):
    _adj = np.array(adj)
    _nums = adj.shape[0]
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(adj.shape[0], np.int)
    return _adj


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-path', default='model/embedding/coco_glove_word2vec_80x300.pkl',
                    type=str, help='Input Path')
parser.add_argument('-a', '--adj-path', default='model/adjacency/coco_adj.pkl',
                    type=str, help='Adjacency Path')
parser.add_argument('-o', '--output-path', default='model/embedding/coco_glove_word2vec_80x300_ec.pkl',
                    type=str, help='Output Path')
parser.add_argument('-n', '--normalise', action='store_true', help='perform normalisation')
parser.add_argument('-ec', '--eigenc', action='store_true', help='perform EC transformation')


def main():
    global args
    args = parser.parse_args()

    with open(args.input_path, 'rb') as finp:
        inp = pickle.load(finp)

    with open(args.adj_path, 'rb') as fadj:
        result = pickle.load(fadj)
        adj = result['adj']

    if args.eigenc:
        print('Eigenvector Centrality Transformation')
        ec = eigenvector_centrality(adj)
        ec = ec * 10  # scale up by 10x
        out = rowmul(inp, ec)
        with open(args.output_path, 'wb') as fout:
            pickle.dump(out, fout, protocol=pickle.HIGHEST_PROTOCOL)
            print("Written to", args.output_path)


if __name__ == '__main__':
    main()
