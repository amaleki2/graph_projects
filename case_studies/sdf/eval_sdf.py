import torch
import tqdm
import trimesh
import numpy as np
from case_studies.sdf.get_data_sdf import vertex_to_proximity_kdtree, compute_edge_features
from src import get_device
from trimesh.proximity import ProximityQuery
import numpy
if numpy.__version__ < '1.20':
    print('numpy version 1.20 is required for this module')

from numpy.lib.stride_tricks import sliding_window_view
from torch_geometric.data import Data, DataLoader



def read_and_process_mesh(obj_in_file, with_rotate_or_scaling=True, with_scaling_to_unit_box=True):
    mesh = trimesh.load(obj_in_file, force='mesh')
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if with_rotate_or_scaling:
        rot_mat = rotate(mesh, return_matrix=True)
    else:
        rot_mat = None

    if with_scaling_to_unit_box:
        s1 = mesh.bounding_box.centroid
        s2 = 2 / np.max(mesh.bounding_box.extents)
        new_vertices = mesh.vertices - s1
        mesh.vertices = new_vertices * s2
    else:
        s1, s2 = None, None

    return mesh, (rot_mat, s1, s2)



# taken from mesh_to_sdf
def get_raster_points(voxel_resolution):
    points = np.meshgrid(
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points


def get_all_edges(all_points, radius, min_n_neghbours, max_n_neighbours):
    import pickle
    with open('tree.pkl', 'rb') as fid:
        tree = pickle.load(fid)
    # tree = KDTree(all_points)
    dist, idx = tree.query(all_points, k=max_n_neighbours)
    s1, s2 = idx.shape
    idx = np.stack((np.tile(np.arange(s1), (s2, 1)).T, idx), axis=2).reshape(-1, 2)  # get list of pairs
    indicator = dist < radius
    if min_n_neghbours is not None:
        indicator[:min_n_neghbours] = True  # set the minimum number of edges
    indicator = indicator.reshape(-1)
    idx = idx[indicator]
    edges = idx.T
    return edges


def get_sub_voxels_indices(voxels_res, sub_voxels_res):
    N, n = voxels_res, sub_voxels_res
    assert N % n == 0
    x = np.arange(N ** 3).reshape((N, N, N))
    m1, m2 = N - n + 1, N // n
    return sliding_window_view(x, (m1, m1, m1))[:, :, :, ::n, ::n, ::n].reshape(n ** 3, m2 ** 3)


def get_grid_points_sfds(mesh, points):
    return -ProximityQuery(mesh).signed_distance(points)


def get_sub_voxels_edges(all_edges, sub_voxel_id):
    sender, receiver = all_edges[np.logical_and(all_edges <= sub_voxel_id.max(), all_edges >= sub_voxel_id.min())]
    sender_in = np.in1d(sender, sub_voxel_id)
    receiver_in = np.in1d(receiver, sub_voxel_id)
    both_in = np.logical_and(sender_in, receiver_in)
    edges = all_edges[:, both_in]
    return edges


def get_edge_features(x, edge_index, include_abs=False):
    e1, e2 = edge_index
    edge_attrs = x[e1, :] - x[e2, :]
    if include_abs:
        edge_attrs= np.concatenate((edge_attrs, np.abs(edge_attrs[:, :2])), axis=1)
    return edge_attrs


def create_voxel_dataset(surface_mesh, voxels_res, sub_voxels_res, radius, min_n_neighbours, max_n_neighbours,
                         with_sdf=True):
    mesh = trimesh.load(surface_mesh)
    surface_points = mesh.vertices
    grid_points = get_raster_points(voxels_res)
    all_points = np.concatenate((surface_points, grid_points))
    all_edges = get_all_edges(all_points, radius, min_n_neighbours, max_n_neighbours)
    grid_points_sfds = get_grid_points_sfds(mesh, grid_points) if with_sdf else None
    sub_voxels_indices = get_sub_voxels_indices(voxels_res, sub_voxels_res)

    graph_data_list = []
    for sub_voxel_id in tqdm.tqdm(sub_voxels_indices):
        sub_voxel_grid_points = grid_points[sub_voxel_id]
        sub_voxel_all_points = np.concatenate((surface_points, sub_voxel_grid_points))
        sub_voxel_edges = get_sub_voxels_edges(all_edges, sub_voxel_id)
        sub_voxel_edge_feats = get_edge_features(all_points, sub_voxel_edges)
        u = np.zeros((1, 1))
        if with_sdf:
            sub_voxel_grid_sdfs = grid_points_sfds[sub_voxel_id]
            sub_voxel_all_sdfs = np.concatenate((np.zeros(len(surface_points)), sub_voxel_grid_sdfs))
            graph_data = Data(x=torch.from_numpy(sub_voxel_all_points).type(torch.float32),
                              y=torch.from_numpy(sub_voxel_all_sdfs).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(sub_voxel_edges).type(torch.long),
                              edge_attr=torch.from_numpy(sub_voxel_edge_feats).type(torch.float32))
        else:
            graph_data = Data(x=torch.from_numpy(sub_voxel_all_points).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(sub_voxel_edges).type(torch.long),
                              edge_attr=torch.from_numpy(sub_voxel_edge_feats).type(torch.float32))
        graph_data_list.append(graph_data)
    data_loader = DataLoader(graph_data_list, batch_size=1)
    return data_loader


def create_voxel_dataset2(surface_mesh, voxels_res, sub_voxels_res, radius, min_n_neighbours, max_n_neighbours,
                          with_sdf=False):
    mesh = trimesh.load(surface_mesh)
    mesh, _ = read_and_process_mesh(mesh, with_rotate_or_scaling=False, with_scaling_to_unit_box=True)
    surface_points = mesh.vertices
    grid_points = get_raster_points(voxels_res)
    sub_voxels_indices = get_sub_voxels_indices(voxels_res, sub_voxels_res)
    graph_data_list = []
    for sub_voxel_id in tqdm.tqdm(sub_voxels_indices):
        sub_voxel_grid_points = grid_points[sub_voxel_id]
        sub_voxel_all_points = np.concatenate((surface_points, sub_voxel_grid_points))
        if with_sdf:
            sub_voxel_grid_sdfs = get_grid_points_sfds(mesh, sub_voxel_grid_points)
            y = np.concatenate((np.zeros(len(surface_points)), sub_voxel_grid_sdfs))
            in_or_out = y < 0
            x = np.concatenate((sub_voxel_all_points, in_or_out.reshape(-1, 1)), axis=1)
            y = y.reshape(-1, 1).astype(float)
        else:
            on_surface = np.concatenate((np.ones(len(surface_points)), np.zeros(len(sub_voxel_grid_points))))
            x = np.concatenate((sub_voxel_all_points, on_surface.reshape(-1, 1)), axis=1)
        x = x.astype(float)

        sub_voxel_edges = vertex_to_proximity_kdtree(sub_voxel_all_points, radius, max_n_neighbours=max_n_neighbours,
                                                     min_n_edges=min_n_neighbours, n_features_to_consider=3)
        sub_voxel_edge_feats = compute_edge_features(x, sub_voxel_edges)
        u = np.zeros((1, 1))
        if with_sdf:
            graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                              y=torch.from_numpy(y).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(sub_voxel_edges).type(torch.long),
                              edge_attr=torch.from_numpy(sub_voxel_edge_feats).type(torch.float32))
        else:
            graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(sub_voxel_edges).type(torch.long),
                              edge_attr=torch.from_numpy(sub_voxel_edge_feats).type(torch.float32))
        graph_data_list.append(graph_data)
    data_loader = DataLoader(graph_data_list, batch_size=1)
    return data_loader


def eval_sdf(model, meshfile, save_name, with_sdf=False, radius=0.3, min_n_neighbours=0, max_n_neighbours=40,
             voxels_res=128, sub_voxels_res=8, loss_funcs=None, use_cpu=False):
    grid_sdfs = np.zeros(voxels_res ** 3)
    sub_voxels_indices = get_sub_voxels_indices(voxels_res, sub_voxels_res)
    data_loader = create_voxel_dataset2(meshfile, voxels_res, sub_voxels_res, radius,
                                        min_n_neighbours, max_n_neighbours, with_sdf=with_sdf)
    device = get_device(use_cpu)
    model = model.to(device=device)
    preds = []
    losses = []
    model.load_state_dict(torch.load("save_dir/model_" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device=device)
            pred = model(data)
            preds.append(pred)
            if loss_funcs is not None:
                loss = [func(pred, data) for func in loss_funcs]
                losses.append(loss)
    for (voxel_id, pred) in zip(sub_voxels_indices, preds):
        grid_sdfs[voxel_id] = pred[1][:-sub_voxels_indices.shape[1]]
    grid_sdfs = grid_sdfs.reshape((voxels_res, voxels_res, voxels_res))
    return grid_sdfs, losses



if __name__ == '__main__':
    from src import EncodeProcessDecode
    surface_mesh = r'D:\data\space_claim_round2\tmp\Polygon_19902e4c-3d59-4fb5-a832-7fcf2215a8a9\geomFiles\geom.obj'
    voxels_res = 128
    sub_voxels_res = 8
    radius = 0.3
    min_n_neighbours = 0
    max_n_neighbours = 40
    save_name = "epd_64_5_0.3_spcm_new_1_knn40"
    n_edge_in, n_edge_out = 4, 1
    n_node_in, n_node_out = 4, 1
    n_global_in, n_global_out = 1, 1
    n_hidden = 64
    n_process = 5
    use_cpu = True
    with_sdf = True
    model = EncodeProcessDecode(n_edge_feat_in=n_edge_in, n_edge_feat_out=n_edge_out,
                                n_node_feat_in=n_node_in, n_node_feat_out=n_node_out,
                                n_global_feat_in=n_global_in, n_global_feat_out=n_global_out,
                                mlp_latent_size=n_hidden, num_processing_steps=n_process,
                                process_weights_shared=True)

    eval_sdf(model, surface_mesh, save_name, radius=radius, with_sdf=with_sdf,
             min_n_neighbours=min_n_neighbours, max_n_neighbours=max_n_neighbours,
             voxels_res=voxels_res, sub_voxels_res=sub_voxels_res, loss_funcs=None, use_cpu=use_cpu)