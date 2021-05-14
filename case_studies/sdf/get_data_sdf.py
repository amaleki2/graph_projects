import os
import tqdm
import torch
import meshio
import numpy as np
from torch_geometric.data import Data, DataLoader, DataListLoader
from case_studies.sdf.util_sdf import (cells_to_edges, vertices_to_proximity, vertex_to_proximity_kdtree,
                                       compute_edge_features, add_reversed_edges, add_self_edges)


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