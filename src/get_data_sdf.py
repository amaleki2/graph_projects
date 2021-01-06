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


def vertices_to_proximity(x, radius):
    dist = distance_matrix(x[:, :2], x[:, :2])
    dist += 2 * radius * np.tril(np.ones_like(dist))  # to avoid duplication later
    edges = np.array(np.where(dist < radius)).T
    return edges


def compute_edge_features(x, edge_index):
    e1, e2 = edge_index
    edge_attrs = x[e1, :] - x[e2, :]
    edge_attrs = np.concatenate((edge_attrs, np.abs(edge_attrs[:, :2])), axis=1)
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
                        edge_method='edge', edge_params=None):
    print("preparing sdf data loader")

    # random splitting into train and test
    random_idx = np.random.permutation(range(i_start, n_objects))
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = []
    test_graph_data_list = []

    for idx, graph_data_list in zip([train_idx, test_idx], [train_graph_data_list, test_graph_data_list]):
        for i in tqdm.tqdm(idx):
            mesh_geom = meshio.read(data_folder + "geom%d.vtk" % i)
            mesh_sdf  = meshio.read(data_folder + "sdf%d.vtk" % i)
            x = mesh_geom.points.copy()
            y = mesh_sdf.points.copy()[:, 2]
            x[:, 2] = (y < 0).astype(float)
            y = y / np.sqrt(8)
            y = y.reshape(-1, 1)

            cells = [x for x in mesh_geom.cells if x.type == 'triangle']
            cells = cells[0].data
            cells = np.array(cells).T

            if edge_method == 'edge':
                edges = cells_to_edges(cells.T)
            elif edge_method == 'proximity':
                radius = edge_params['radius']
                edges = vertices_to_proximity(x, radius)
            elif edge_method == 'both':
                edges1 = cells_to_edges(cells.T)
                radius = edge_params['radius']
                edges2 = vertices_to_proximity(x, radius)
                edges = np.concatenate((edges1, edges2), axis=0)
                edge_params['e1'] = len(edges1)  # will be used later for edge_attr
            else:
                raise(NotImplementedError("method %s is not recognized" % edge_method))
            edges = edges.T

            if not reversed_edge_already_included:
                edges = add_reversed_edges(edges)
            if not self_edge_already_included:
                edges = add_self_edges(edges)

            edge_feats = compute_edge_features(x, edges)

            cent = np.mean(x[x[:, 2] == 1, :2], axis=0, keepdims=True)
            area = np.mean(x[:, :2], keepdims=True)
            u = np.concatenate((cent, area), axis=1)

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
