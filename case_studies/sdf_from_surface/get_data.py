import os
import tqdm
import torch
import meshio
import pyflann
import numpy as np
from torch_geometric.data import Data, DataLoader
from scipy.spatial import distance_matrix
from case_studies.sdf.get_data_sdf import cells_to_edges, add_self_edges, add_reversed_edges


def compute_edge_features(x, edge_index):
    e1, e2 = edge_index
    edge_attrs = x[e1, :] - x[e2, :]
    return edge_attrs


def get_points_from_sdf_mesh(mesh):
    points = mesh.points
    sdfs = mesh.point_data['SDF']
    surface_points = points[sdfs == 0]
    volume_points  = points[sdfs != 0]
    volume_sdfs    = sdfs[sdfs != 0]
    return surface_points, volume_points, volume_sdfs


def get_data_loader(n_objects, data_folder, batch_size, eval_frac=0.2, i_start=0):
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
            surface_points, volume_points, volume_sdfs = get_points_from_sdf_mesh(mesh_sdf)
            x = surface_points.astype(float)
            y = volume_points.astype(float)
            z = volume_sdfs.astype(float)
            cells = [x for x in mesh_sdf.cells if x.type == 'triangle']
            cells = cells[0].data.astype(int)
            cells = np.array(cells).T
            edges = cells_to_edges(cells.T)
            edges = edges.T

            edges = add_reversed_edges(edges)
            edges = add_self_edges(edges)
            edges = np.unique(edges, axis=1)   # remove repeated edges
            edge_feats = compute_edge_features(x, edges)

            u = np.zeros((1, 1))

            graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                              y=torch.from_numpy(y).type(torch.float32),
                              z=torch.from_numpy(z).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(edges).type(torch.long),
                              edge_attr=torch.from_numpy(edge_feats).type(torch.float32)
                              )
            graph_data_list.append(graph_data)
    train_data = DataLoader(train_graph_data_list, batch_size=batch_size)
    test_data = DataLoader(test_graph_data_list, batch_size=batch_size)
    return train_data, test_data