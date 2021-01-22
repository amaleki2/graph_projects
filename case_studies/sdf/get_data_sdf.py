import os
import tqdm
import torch
import meshio
import numpy as np
from torch_geometric.data import Data, DataLoader
from scipy.spatial import distance_matrix


def cells_to_edges(cells):
    v0v1 = cells[:, :2]
    v1v2 = cells[:, 1:]
    v0v2 = cells[:, 0:3:2]
    edge_pairs = np.concatenate((v0v1, v1v2, v0v2))
    edge_pairs = np.sort(edge_pairs, axis=1)
    edge_pairs = np.unique(edge_pairs, axis=0)
    return edge_pairs


def vertices_to_proximity(x, radius, cache_knn=None):
    if cache_knn is not None and os.path.isfile(cache_knn):
        dist_val, dist_idx = np.load(cache_knn)
        dist_idx = dist_idx.astype(int)
    else:
        dist = distance_matrix(x[:, :2], x[:, :2])
        #dist += dist.max() * np.tril(np.ones_like(dist))  # to avoid duplication later
        dist_idx = np.argsort(dist, axis=1)[:, :25]
        dist_val = np.sort(dist, axis=1)[:, :25]
        if cache_knn is not None:
            np.save(cache_knn, np.array([dist_val, dist_idx]))
    neighbours_idx = np.where(dist_val < radius)
    edges = [neighbours_idx[0], dist_idx[neighbours_idx[0], neighbours_idx[1]]]
    edges = np.array(edges).T
    return edges


def compute_edge_features(x, edge_index, include_abs=False):
    e1, e2 = edge_index
    edge_attrs = x[e1, :] - x[e2, :]
    if include_abs:
        edge_attrs= np.concatenate((edge_attrs, np.abs(edge_attrs[:, :2])), axis=1)
    return edge_attrs


def add_reversed_edges(edges):
    edges_reversed = np.flipud(edges)
    edges = np.concatenate([edges, edges_reversed], axis=1)
    return edges


def add_self_edges(edges):
    n_nodes = edges.max() + 1
    self_edges = [list(range(n_nodes))] * 2
    self_edges = np.array(self_edges)
    edges = np.concatenate([edges, self_edges], axis=1)
    return edges


def get_sdf_data_loader(n_objects, data_folder, batch_size, eval_frac=0.2, i_start=0,
                        reversed_edge_already_included=False, self_edge_already_included=False,
                        edge_method='edge', edge_params=None, no_global=False, with_vertices=False):
    # random splitting into train and test
    random_idx = np.random.permutation(range(i_start, n_objects))
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = []
    test_graph_data_list = []

    for idx, graph_data_list in zip([train_idx, test_idx], [train_graph_data_list, test_graph_data_list]):
        for i in tqdm.tqdm(idx):
            mesh_file = data_folder + "sdf%d.vtk" % i
            mesh_sdf = meshio.read(mesh_file)
            x = mesh_sdf.points.copy()
            y = mesh_sdf.points.copy()[:, 2]
            x[:, 2] = y < 0
            x = x.astype(float)
            if with_vertices:
                x = np.hstack((x, np.zeros((len(x), 1))))
                vertices_file = mesh_file.replace('vtk', 'npy')
                vertices = np.load(vertices_file)
                for v in vertices:
                    idx = np.isclose(x[:, :2], v, atol=1e-8, rtol=1e-8).all(axis=1)
                    x[idx, 3] = 1
                # is_vertex = (np.min(distance_matrix(x[:, :2], vertices), axis=1, keepdims=True) < 1e-3).astype(int)
                # assert is_vertex.sum() == len(vertices)
                # x = np.concatenate((x, is_vertex), axis=1)
                # if x[:, 3].sum() != len(vertices):
                #     print("kir khar")
                assert x[:, 3].sum() == len(vertices)
            y = y / np.sqrt(8)
            y = y.reshape(-1, 1)

            cells = [x for x in mesh_sdf.cells if x.type == 'triangle']
            cells = cells[0].data.astype(int)
            cells = np.array(cells).T

            if edge_method == 'edge':
                edges = cells_to_edges(cells.T)
            elif edge_method == 'proximity':
                knn_idx = data_folder + "knn%d.npy" % i
                radius = edge_params['radius']
                edges = vertices_to_proximity(x, radius, cache_knn=knn_idx)
            elif edge_method == 'both':
                edges1 = cells_to_edges(cells.T)
                radius = edge_params['radius']
                knn_idx = data_folder + "knn%d.npy" % i
                edges2 = vertices_to_proximity(x, radius, cache_knn=knn_idx)
                edges = np.concatenate((edges1, edges2), axis=0)
            else:
                raise(NotImplementedError("method %s is not recognized" % edge_method))
            edges = edges.T

            if not reversed_edge_already_included:
                edges = add_reversed_edges(edges)
            if not self_edge_already_included:
                edges = add_self_edges(edges)
            edges = np.unique(edges, axis=1)   # remove repeated edges
            edge_feats = compute_edge_features(x, edges)

            if not no_global:
                cent = np.mean(x[x[:, 2] == 1, :2], axis=0, keepdims=True)
                area = np.mean(x[:, :2], keepdims=True)
                u = np.concatenate((cent, area), axis=1)
            else:
                u = np.zeros((1, 1))

            graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                              y=torch.from_numpy(y).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(edges).type(torch.long),
                              edge_attr=torch.from_numpy(edge_feats).type(torch.float32),
                              face=torch.from_numpy(cells).type(torch.long))
            graph_data_list.append(graph_data)
    train_data = DataLoader(train_graph_data_list, batch_size=batch_size)
    test_data = DataLoader(test_graph_data_list, batch_size=batch_size)
    return train_data, test_data
