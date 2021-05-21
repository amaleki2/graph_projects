import tqdm
import torch
import meshio
import trimesh
import numpy as np
from trimesh.proximity import ProximityQuery
from scipy.interpolate import griddata

from src import get_device
from case_studies.sdf.surface_mesh_utils import generate_surface_mesh
from numpy.lib.stride_tricks import sliding_window_view
from torch_geometric.data import Data, DataLoader, DataListLoader
from case_studies.sdf.util_sdf import (vertex_to_proximity_kdtree, compute_edge_features, add_reversed_edges)
from joblib import Parallel, delayed


def read_and_process_mesh(obj_in_file, with_rotate_or_scaling=True, with_scaling_to_unit_box=True, scaler=2):
    mesh = trimesh.load(obj_in_file, force='mesh')
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if with_rotate_or_scaling:
        rot_mat = rotate(mesh, return_matrix=True)
    else:
        rot_mat = None

    if with_scaling_to_unit_box:
        s1 = mesh.bounding_box.centroid
        s2 = scaler / np.max(mesh.bounding_box.extents)
        new_vertices = mesh.vertices - s1
        mesh.vertices = new_vertices * s2
    else:
        s1, s2 = None, None

    return mesh, (rot_mat, s1, s2)


def get_surface_points(mesh, method='mesh', mesh_size=0.05, show=False):
    if method == 'mesh':
        generate_surface_mesh(mesh_points=mesh.vertices, mesh_faces=mesh.faces, lc=mesh_size,
                              saved_name='tmp.vtk', show=show)
        surface_mesh = meshio.read('tmp.vtk')
        surface_points = surface_mesh.points
        surface_faces = surface_mesh.get_cells_type('triangle')
        return surface_points, surface_faces


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


def get_all_edges(all_points, radius, min_n_neighbours, max_n_neighbours):
    import pickle
    with open('tree.pkl', 'rb') as fid:
        tree = pickle.load(fid)
    # tree = KDTree(all_points)
    dist, idx = tree.query(all_points, k=max_n_neighbours)
    s1, s2 = idx.shape
    idx = np.stack((np.tile(np.arange(s1), (s2, 1)).T, idx), axis=2).reshape(-1, 2)  # get list of pairs
    indicator = dist < radius
    if min_n_neighbours is not None:
        indicator[:min_n_neighbours] = True  # set the minimum number of edges
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


def create_voxel_dataset(surface_mesh, voxels_res, sub_voxels_res, edge_params, n_jobs=1, scaler=2,
                         batch_size=1, include_reverse_edges=False, data_parallel=False, force_n_volume_points_to=None):
    mesh = trimesh.load(surface_mesh, force='mesh', skip_materials=True)
    mesh, _ = read_and_process_mesh(mesh, with_rotate_or_scaling=False, with_scaling_to_unit_box=True, scaler=scaler)
    surface_points, _ = get_surface_points(mesh, mesh_size=0.1, show=False)
    grid_points = get_raster_points(voxels_res)
    sub_voxels_indices = get_sub_voxels_indices(voxels_res, sub_voxels_res)

    def func(id):
        sub_voxel_grid_points = grid_points[id]
        if force_n_volume_points_to is not None:
            n_missing = force_n_volume_points_to - len(sub_voxel_grid_points)
            rnd_points = np.random.random((n_missing, 3)) * 2 - 1
            sub_voxel_grid_points = np.concatenate((rnd_points, sub_voxel_grid_points))
        sub_voxel_all_points = np.concatenate((surface_points, sub_voxel_grid_points))
        sub_voxel_grid_sdfs = get_grid_points_sfds(mesh, sub_voxel_grid_points)
        y = np.concatenate((np.zeros(len(surface_points)), sub_voxel_grid_sdfs))
        on_surface = np.concatenate((np.ones(len(surface_points)), np.zeros(len(sub_voxel_grid_points))))
        x = np.concatenate((sub_voxel_all_points, on_surface.reshape(-1, 1)), axis=1)
        x = x.astype(float)
        y = y.reshape(-1, 1).astype(float)

        sub_voxel_edges = vertex_to_proximity_kdtree(sub_voxel_all_points, edge_params, n_features_to_consider=3)
        if include_reverse_edges:
            sub_voxel_edges = add_reversed_edges(sub_voxel_edges)
        sub_voxel_edges = np.unique(sub_voxel_edges, axis=1)

        sub_voxel_edge_feats = compute_edge_features(x, sub_voxel_edges)
        u = np.zeros((1, 1))
        graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                          y=torch.from_numpy(y).type(torch.float32),
                          u=torch.from_numpy(u).type(torch.float32),
                          edge_index=torch.from_numpy(sub_voxel_edges).type(torch.long),
                          edge_attr=torch.from_numpy(sub_voxel_edge_feats).type(torch.float32))
        return graph_data

    if n_jobs == 1:
        graph_data_list = [func(id) for id in tqdm.tqdm(sub_voxels_indices)]
    else:
        graph_data_list = Parallel(n_jobs=n_jobs)(delayed(func)(idx) for idx in tqdm.tqdm(sub_voxels_indices))
    # graph_data_list = []
    # for sub_voxel_id in tqdm.tqdm(sub_voxels_indices):
    #     sub_voxel_grid_points = grid_points[sub_voxel_id]
    #     sub_voxel_all_points = np.concatenate((surface_points, sub_voxel_grid_points))
    #     sub_voxel_grid_sdfs = get_grid_points_sfds(mesh, sub_voxel_grid_points)
    #     y = np.concatenate((np.zeros(len(surface_points)), sub_voxel_grid_sdfs))
    #     on_surface = np.concatenate((np.ones(len(surface_points)), np.zeros(len(sub_voxel_grid_points))))
    #     x = np.concatenate((sub_voxel_all_points, on_surface.reshape(-1, 1)), axis=1)
    #     x = x.astype(float)
    #     y = y.reshape(-1, 1).astype(float)
    #
    #     sub_voxel_edges = vertex_to_proximity_kdtree(sub_voxel_all_points, edge_params, n_features_to_consider=3)
    #     if include_reverse_edges:
    #         sub_voxel_edges = add_reversed_edges(sub_voxel_edges)
    #     sub_voxel_edges = np.unique(sub_voxel_edges, axis=1)
    #
    #     sub_voxel_edge_feats = compute_edge_features(x, sub_voxel_edges)
    #     u = np.zeros((1, 1))
    #     graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
    #                       y=torch.from_numpy(y).type(torch.float32),
    #                       u=torch.from_numpy(u).type(torch.float32),
    #                       edge_index=torch.from_numpy(sub_voxel_edges).type(torch.long),
    #                       edge_attr=torch.from_numpy(sub_voxel_edge_feats).type(torch.float32))
    #     graph_data_list.append(graph_data)

    if data_parallel:
        data_loader = DataListLoader(graph_data_list, batch_size=batch_size)
    else:
        data_loader = DataLoader(graph_data_list, batch_size=batch_size)
    return data_loader, sub_voxels_indices


def create_random_dataset(surface_mesh, n_volume_points, radius, min_n_neighbours, max_n_neighbours,
                          with_sdf=False, batch_size=1):
    mesh = trimesh.load(surface_mesh)
    mesh, _ = read_and_process_mesh(mesh, with_rotate_or_scaling=False, with_scaling_to_unit_box=True)

    surface_points, _ = get_surface_points(mesh, mesh_size=0.1, show=False)
    n_splits = round(n_volume_points // 5000)
    volume_points = np.random.random((n_volume_points, 3)) * 2 - 1
    volume_points_split = np.array_split(volume_points, n_splits)
    graph_data_list = []
    for points in tqdm.tqdm(volume_points_split):
        all_points = np.concatenate((surface_points, points))
        if with_sdf:
            sub_voxel_grid_sdfs = get_grid_points_sfds(mesh, points)
            y = np.concatenate((np.zeros(len(surface_points)), sub_voxel_grid_sdfs))
            in_or_out = y < 0
            x = np.concatenate((all_points, in_or_out.reshape(-1, 1)), axis=1)
            y = y.reshape(-1, 1).astype(float)
        else:
            on_surface = np.concatenate((np.ones(len(surface_points)), np.zeros(len(points))))
            x = np.concatenate((all_points, on_surface.reshape(-1, 1)), axis=1)
        x = x.astype(float)

        edges = vertex_to_proximity_kdtree(all_points, radius,
                                           max_n_neighbours=max_n_neighbours,
                                           min_n_edges=min_n_neighbours,
                                           n_features_to_consider=3)
        sub_voxel_edge_feats = compute_edge_features(x, edges)
        u = np.zeros((1, 1))
        if with_sdf:
            graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                              y=torch.from_numpy(y).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(edges).type(torch.long),
                              edge_attr=torch.from_numpy(sub_voxel_edge_feats).type(torch.float32))
        else:
            graph_data = Data(x=torch.from_numpy(x).type(torch.float32),
                              u=torch.from_numpy(u).type(torch.float32),
                              edge_index=torch.from_numpy(edges).type(torch.long),
                              edge_attr=torch.from_numpy(sub_voxel_edge_feats).type(torch.float32))
        graph_data_list.append(graph_data)
    data_loader = DataLoader(graph_data_list, batch_size=batch_size)
    # data_loader = DataListLoader(graph_data_list, batch_size=batch_size)
    return data_loader, volume_points_split


def eval_sdf(model, data_loader, save_name, loss_funcs=None, device=torch.device('cuda')):
    preds = []
    losses = []
    data_parallel = isinstance(device, list)
    if data_parallel:  # data parallel
        device0 = torch.device('cuda:%d'%device[0])
        model.load_state_dict(torch.load("save_dir/model_" + save_name + ".pth", map_location=device0))
        model = model.to(device0)
    else:
        model.load_state_dict(torch.load("save_dir/model_" + save_name + ".pth", map_location=device))
        model = model.to(device=device)
    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(data_loader):
            if not data_parallel:
                data = data.to(device=device)
            pred = model(data)
            preds.append(pred)
            if loss_funcs is not None:
                loss = [func(pred, data) for func in loss_funcs]
                losses.append(loss)
    return preds, losses


def sdf_to_grids(preds, voxels_res, sub_voxels_indices, batch_size=1):
    assert len(preds) * batch_size == len(sub_voxels_indices)
    grid_sdfs = np.zeros(voxels_res ** 3)
    sdf_preds = np.array([p[1].cpu().numpy().squeeze() for p in preds])
    sdf_preds = sdf_preds.reshape(-1, batch_size)
    sdf_preds = sdf_preds[:, -sub_voxels_indices.shape[1]:]
    for (id, sp) in zip(sub_voxels_indices, sdf_preds):
        grid_sdfs[id] = sp
    grid_sdfs = grid_sdfs.reshape((voxels_res, voxels_res, voxels_res))
    return grid_sdfs


def sdf_to_grids_interpolate(volume_points, preds, method='linear', res=128):
    eps = 1e-3
    res_complex = res * 1j
    grid_x, grid_y, grid_z = np.mgrid[-1+eps:1-eps:res_complex, -1+eps:1-eps:res_complex, -1+eps:1-eps:res_complex]
    points = np.concatenate(volume_points)
    preds  = np.concatenate([p[1][-len(vp):].cpu().numpy().squeeze() for (p, vp) in zip(preds, volume_points)])
    grid_sdfs = griddata(points, preds, (grid_x, grid_y, grid_z), method=method)
    return grid_sdfs


if __name__ == '__main__':
    from src import EncodeProcessDecode
    surface_mesh = r'C:\Users\amaleki\Downloads\NewMesh\Servo Horn - Half Arm_20.obj'
    voxels_res = 16
    sub_voxels_res = 4
    radius = 0.3
    min_n_edges = 0
    max_n_nedges = 40
    save_name = "epd_64_5_0.3_spcm_new_1_knn40"
    n_edge_in, n_edge_out = 4, 1
    n_node_in, n_node_out = 4, 1
    n_global_in, n_global_out = 1, 1
    n_hidden = 64
    n_process = 5
    device = '0'
    device = get_device(device)
    data_parallel = isinstance(device, list)
    model = EncodeProcessDecode(n_edge_feat_in=n_edge_in, n_edge_feat_out=n_edge_out,
                                n_node_feat_in=n_node_in, n_node_feat_out=n_node_out,
                                n_global_feat_in=n_global_in, n_global_feat_out=n_global_out,
                                mlp_latent_size=n_hidden, num_processing_steps=n_process,
                                process_weights_shared=True)

    edge_params = {'radius': radius, 'min_n_edges': min_n_edges, 'max_n_edges': max_n_nedges}
    import time
    t1 = time.time()
    data_loader, sub_voxels_indices = create_voxel_dataset(surface_mesh, voxels_res, sub_voxels_res, edge_params,
                                                           batch_size=1, include_reverse_edges=True,
                                                           data_parallel=data_parallel, n_jobs=6)
    t2 = time.time()
    print(t2 - t1)

    preds, losses = eval_sdf(model, data_loader, save_name, loss_funcs=None, device=device)

    grid_sdfs = sdf_to_grids(preds, voxels_res, sub_voxels_indices)