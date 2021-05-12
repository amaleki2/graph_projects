import os
import tqdm
import torch
import meshio
import pyflann
import numpy as np
from torch_geometric.data import Data, DataLoader, DataListLoader
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree


def cells_to_edges(cells):
    v0v1 = cells[:, :2]
    v1v2 = cells[:, 1:]
    v0v2 = cells[:, 0:3:2]
    edge_pairs = np.concatenate((v0v1, v1v2, v0v2))
    edge_pairs = np.sort(edge_pairs, axis=1)
    edge_pairs = np.unique(edge_pairs, axis=0)
    return edge_pairs


def vertices_to_proximity(x, radius, cache_knn=None, max_n_neighbours=40, approx_knn=False, min_n_edges=None):
    if cache_knn is not None and os.path.isfile(cache_knn):
        dist_val, dist_idx = np.load(cache_knn)
        dist_idx = dist_idx.astype(int)
    else:
        n_features_to_consider = min(3, x.shape[1] - 1)  # 2 for 2d, 3 for 3d. ignore normal features.
        if approx_knn and len(x) > 10000:
            flann = pyflann.FLANN()
            dist_idx, dist_val = flann.nn(x[:, :n_features_to_consider], x[:, :n_features_to_consider],
                                          max_n_neighbours, algorithm="kmeans", branching=32, iterations=7, checks=16)
            dist_val **= 0.5
        else:
            dist = distance_matrix(x[:, :n_features_to_consider], x[:, :n_features_to_consider])
            dist_idx = np.argsort(dist, axis=1)[:, :max_n_neighbours]
            dist_val = np.sort(dist, axis=1)[:, :max_n_neighbours]
        if cache_knn is not None:
            np.save(cache_knn, np.array([dist_val, dist_idx]))
    neighbours_idx = np.where(dist_val < radius)
    senders = neighbours_idx[0].astype(np.int32)
    receivers = neighbours_idx[1].astype(np.int32)
    edges = [senders, dist_idx[senders, receivers]]
    edges = np.array(edges)
    if min_n_edges is not None:
        new_edges = [np.arange(len(x)).repeat(min_n_edges), dist_idx[:, :min_n_edges].reshape(-1)]
        new_edges = np.array(new_edges).astype(int)
        edges = np.concatenate((edges, new_edges), axis=1)
    edges = edges.T
    return edges


def vertex_to_proximity_kdtree(x, radius, max_n_neighbours=40, min_n_edges=0, n_features_to_consider=3):
    points = x[:, :n_features_to_consider]
    tree = KDTree(points)
    dist, idx = tree.query(points, k=max_n_neighbours)
    s1, s2 = idx.shape
    idx = np.stack((np.tile(np.arange(s1), (s2, 1)).T, idx), axis=2).reshape(-1, 2)  # get list of pairs
    indicator = dist < radius
    indicator[:min_n_edges] = 1   # set the minimum number of edges
    indicator = indicator.reshape(-1)
    idx = idx[indicator]  # set the radius of proximity
    edges = idx.T
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
                        edge_method='edge', edge_params=None, no_global=False, no_edge=False):
    # random splitting into train and test
    random_idx = np.random.permutation(range(i_start, n_objects))
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = []
    test_graph_data_list = []

    for idx, graph_data_list in zip([train_idx, test_idx], [train_graph_data_list, test_graph_data_list]):
        for i in tqdm.tqdm(idx):
            mesh_file = os.path.join(data_folder, "sdf%d.vtk" % i)
            mesh_sdf = meshio.read(mesh_file)
            x = mesh_sdf.points.copy()
            y = mesh_sdf.points.copy()[:, 2:]
            x[:, 2] = x[:, 2] < 0
            x = x.astype(float)
            y = y.astype(float)

            cells = [x for x in mesh_sdf.cells if x.type == 'triangle']
            cells = cells[0].data.astype(int)
            cells = np.array(cells).T

            if edge_method == 'edge':
                edges = cells_to_edges(cells.T)
            elif edge_method == 'proximity':
                knn_idx = os.path.join(data_folder, "knn%d.npy" % i)
                radius = edge_params['radius']
                edges = vertices_to_proximity(x, radius, cache_knn=knn_idx)
            elif edge_method == 'both':
                edges1 = cells_to_edges(cells.T)
                radius = edge_params['radius']
                knn_idx = os.path.join(data_folder, "knn%d.npy" % i)
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
            if no_edge:
                edge_feats = np.zeros((edges.shape[1], 1))
            else:
                edge_feats = compute_edge_features(x, edges)

            if not no_global:
                cent = np.mean(x[x[:, 2] == 1, :2], axis=0, keepdims=True)
                area = np.mean(x[:, 2:], keepdims=True)
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


def get_sdf_data_loader_3d(n_objects, data_folder, batch_size, eval_frac=0.2, i_start=0,
                           reversed_edge_already_included=False, self_edge_already_included=False,
                           edge_method='edge', edge_params=None, no_global=False, no_edge=False,
                           with_normals=False, with_sdf_signs=True, data_parallel=False, shuffle=False):
    # random splitting into train and test
    random_idx = np.random.permutation(range(i_start, n_objects))
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = []
    test_graph_data_list = []

    for idx, graph_data_list in zip([train_idx, test_idx], [train_graph_data_list, test_graph_data_list]):
        for i in tqdm.tqdm(idx):
            mesh_file = os.path.join(data_folder, "sdf%d.vtk" % i)
            mesh_sdf = meshio.read(mesh_file)
            x = mesh_sdf.points
            if with_normals:
                normals = mesh_sdf.point_data['NORMALS']
                x = np.concatenate((x, normals), axis=1)
            y = mesh_sdf.point_data['SDF']
            if with_sdf_signs:
                in_or_out = y < 0
                x = np.concatenate((x, in_or_out.reshape(-1, 1)), axis=1)
            else:
                on_surface = y == 0
                x = np.concatenate((x, on_surface.reshape(-1, 1)), axis=1)
            x = x.astype(float)
            y = y.reshape(-1, 1).astype(float)

            cells = [x for x in mesh_sdf.cells if x.type == 'triangle']
            cells = cells[0].data.astype(int)
            cells = np.array(cells).T

            if edge_method == 'edge':
                raise(ValueError("edge method is not correct for 3d"))
            elif edge_method == 'proximity':
                radius = edge_params['radius']
                min_n_edges = edge_params.get('min_n_edges', 0)
                max_n_neighbours = edge_params.get('max_n_neighbours', 40)
                # knn_idx = os.path.join(data_folder, "knn%d.npy" % i)
                # edges = vertices_to_proximity(x, radius, cache_knn=knn_idx, min_n_edges=min_n_edges)
                # edges = edges.T
                edges = vertex_to_proximity_kdtree(x, radius, max_n_neighbours=max_n_neighbours,
                                                   min_n_edges=min_n_edges, n_features_to_consider=3)
            elif edge_method == 'both':
                edges1 = cells_to_edges(cells.T)
                radius = edge_params['radius']
                min_n_edges = edge_params.get('min_n_edges')
                knn_idx = os.path.join(data_folder, "knn%d.npy" % i)
                edges2 = vertices_to_proximity(x, radius, cache_knn=knn_idx, min_n_edges=min_n_edges)
                edges = np.concatenate((edges1, edges2), axis=0)
            else:
                raise(NotImplementedError("method %s is not recognized" % edge_method))
            # edges = edges.T

            if not reversed_edge_already_included:
                edges = add_reversed_edges(edges)
            if not self_edge_already_included:
                edges = add_self_edges(edges)
            edges = np.unique(edges, axis=1)   # remove repeated edges

            if no_edge:
                edge_feats = np.zeros((len(edges.T), 1))
            else:
                edge_feats = compute_edge_features(x, edges)

            if not no_global:
                cent = np.mean(x[x[:, 3] == 1, :3], axis=0, keepdims=True)
                area = np.mean(x[:, 3:], keepdims=True)
                u = np.concatenate((cent, area), axis=1)
            else:
                u = np.zeros((1, 1))

            graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                              y=torch.from_numpy(y).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(edges).type(torch.long),
                              edge_attr=torch.from_numpy(edge_feats).type(torch.float32),
                              #face=torch.from_numpy(cells).type(torch.long)
                              )
            graph_data_list.append(graph_data)
    if data_parallel:
        train_data = DataListLoader(train_graph_data_list, batch_size=batch_size, shuffle=shuffle)
        test_data = DataListLoader(test_graph_data_list, batch_size=batch_size, shuffle=shuffle)
    else:
        train_data = DataLoader(train_graph_data_list, batch_size=batch_size, shuffle=shuffle)
        test_data = DataLoader(test_graph_data_list, batch_size=batch_size, shuffle=shuffle)
    return train_data, test_data