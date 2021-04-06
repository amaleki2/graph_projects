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
        dist = distance_matrix(x[:, :-1], x[:, :-1])
        #dist += dist.max() * np.tril(np.ones_like(dist))  # to avoid duplication later
        dist_idx = np.argsort(dist, axis=1)[:, :25]
        dist_val = np.sort(dist, axis=1)[:, :25]
        if cache_knn is not None:
            np.save(cache_knn, np.array([dist_val, dist_idx]))
    neighbours_idx = np.where(dist_val < radius)
    senders = neighbours_idx[0].astype(np.int32)
    receivers = neighbours_idx[1].astype(np.int32)
    edges = [senders, dist_idx[senders, receivers]]
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


def process_mesh(mesh, edge_method='proximity', data_folder=None, edge_params=None, knn_file=None,
                 reversed_edge_already_included=False, self_edge_already_included=False, no_global=False):
    x = mesh.points.copy()[:, :2]
    cells = [x for x in mesh.cells if x.type == 'triangle']
    cells = cells[0].data.astype(int)
    cells = np.array(cells).T

    if edge_method == 'edge':
        edges = cells_to_edges(cells.T)
    elif edge_method == 'proximity':
        knn_idx = os.path.join(data_folder, knn_file)
        radius = edge_params['radius']
        edges = vertices_to_proximity(x, radius, cache_knn=knn_idx)
    elif edge_method == 'both':
        edges1 = cells_to_edges(cells.T)
        radius = edge_params['radius']
        knn_idx = os.path.join(data_folder, knn_file)
        edges2 = vertices_to_proximity(x, radius, cache_knn=knn_idx)
        edges = np.concatenate((edges1, edges2), axis=0)
    else:
        raise (NotImplementedError("method %s is not recognized" % edge_method))
    edges = edges.T

    if not reversed_edge_already_included:
        edges = add_reversed_edges(edges)
    if not self_edge_already_included:
        edges = add_self_edges(edges)
    edges = np.unique(edges, axis=1)  # remove repeated edges
    edge_feats = compute_edge_features(x, edges)

    if not no_global:
        u = np.mean(x, keepdims=True)
    else:
        u = np.zeros((1, 1))
    graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                      u=torch.from_numpy(u).type(torch.float32),
                      edge_index=torch.from_numpy(edges).type(torch.long),
                      edge_attr=torch.from_numpy(edge_feats).type(torch.float32),
                      face=torch.from_numpy(cells).type(torch.long))
    return graph_data


def get_mesh_morphing_data_loader(n_objects, data_folder, batch_size, eval_frac=0.2, i_start=0,
                                  reversed_edge_already_included=False, self_edge_already_included=False,
                                  edge_method='edge', edge_params=None, no_global=False):
    # random splitting into train and test
    random_idx = np.random.permutation(range(i_start, n_objects))
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = []
    test_graph_data_list = []

    for idx, graph_data_list in zip([train_idx, test_idx], [train_graph_data_list, test_graph_data_list]):
        for i in tqdm.tqdm(idx):
            mesh_file_in = os.path.join(data_folder, "mesh_%d.vtk" % i)
            knn_in = "knn_%d.npy" % i
            mesh_in = meshio.read(mesh_file_in)
            graph_in = process_mesh(mesh_in, edge_method=edge_method, data_folder=data_folder, edge_params=edge_params,
                                    knn_file=knn_in, reversed_edge_already_included=reversed_edge_already_included,
                                    self_edge_already_included=self_edge_already_included, no_global=no_global)
            mesh_file_out = os.path.join(data_folder, "mesh_perturbed_%d.vtk" % i)
            knn_out = "knn_perturbed_%d.npy" % i
            mesh_out = meshio.read(mesh_file_out)
            graph_out = process_mesh(mesh_out, edge_method=edge_method, data_folder=data_folder, edge_params=edge_params,
                                    knn_file=knn_out, reversed_edge_already_included=reversed_edge_already_included,
                                    self_edge_already_included=self_edge_already_included, no_global=no_global)
            graph_data_list.append((graph_in, graph_out))
    train_data = DataLoader(train_graph_data_list, batch_size=batch_size)
    test_data = DataLoader(test_graph_data_list, batch_size=batch_size)
    return train_data, test_data