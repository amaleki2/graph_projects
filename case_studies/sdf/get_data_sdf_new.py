import os
import tqdm
import glob
import gmsh
import torch
import shutil
import meshio
import trimesh
import numpy as np
from trimesh.proximity import ProximityQuery
from torch_geometric.data import Data, DataLoader, DataListLoader
from case_studies.sdf.util_sdf import (cells_to_edges, vertices_to_proximity, vertex_to_proximity_kdtree,
                                       compute_edge_features, add_reversed_edges, add_self_edges)
from case_studies.sdf.surface_mesh_utils import rotate, refine_surface_mesh


def read_process_and_save_geometry(obj_in_file, obj_out_file, obj_out_file_refined,
                                   merge_vertex=True,
                                   with_random_rotation=True,
                                   with_scaling_to_unit_box=True,
                                   mesh_refine_size=0.1,
                                   max_num_nodes=3000,
                                   show=False):

    mesh = trimesh.load(obj_in_file, force='mesh')
    if mesh.vertices.shape[0] > max_num_nodes:
        print('maximum number of node')
        return False

    if merge_vertex:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)

    if not mesh.is_watertight:
        print('mesh not watertight')
        return False

    if with_random_rotation:
        rotate(mesh)

    if with_scaling_to_unit_box:
        s1 = mesh.bounding_box.centroid
        s2 = 2 / np.max(mesh.bounding_box.extents)
        new_vertices = mesh.vertices - s1
        mesh.vertices = new_vertices * s2

    try:
        refined_mesh = refine_surface_mesh(mesh, mesh_size=mesh_refine_size, show=show)
    except:
        print('gmsh errored in file %s'%obj_in_file)
        gmsh.finalize()
        return False

    with open(obj_out_file, 'w') as fid:
        mesh.export(fid, file_type='obj')

    with open(obj_out_file_refined, 'w') as fid:
        refined_mesh.export(fid, file_type='obj')

    return True


def get_volume_points_randomly(n_points):
    return np.random.random((n_points, 3)) * 2 - 1


def get_sdf_points(mesh, points):
    return - ProximityQuery(mesh).signed_distance(points)


def write_to_file(surface_points, volume_points, volume_sdfs, surface_face, vtk_out_file, volume_faces=None):
    if volume_faces is not None:
        surface_face = np.concatenate((surface_face, volume_faces))
    cells = [("triangle", surface_face)]
    all_points = np.concatenate((surface_points, volume_points))
    all_sdfs = np.concatenate((np.zeros(len(surface_points)), volume_sdfs))
    mesh = meshio.Mesh(all_points, cells, point_data={"SDF": all_sdfs})
    mesh.write(vtk_out_file)


def prepare_processed_cad_data_folder(cad_data_folder,
                                      processed_cad_data_folder=None,
                                      data_format=None,
                                      with_random_rotation=True,
                                      with_scaling_to_unit_box=True,
                                      mesh_refine_size=0.1,
                                      show_refined_mesh=False):
    if processed_cad_data_folder is None:
        parent_folder, folder_name = os.path.split(cad_data_folder)
        processed_folder_name = folder_name + "_processed"
        processed_cad_data_folder = os.path.join(parent_folder, processed_folder_name)

    if not os.path.isdir(processed_cad_data_folder):
        os.makedirs(processed_cad_data_folder)

    if data_format is None:
        cad_files_list = [os.path.join(cad_data_folder, f) for f in os.path.join(cad_data_folder)]
    else:
        cad_files_list = list(glob.iglob(cad_data_folder + data_format))

    id = 0
    for cad_file in cad_files_list:
        processed_cad_file = os.path.join(processed_cad_data_folder, "processed_mesh_%d.obj"%id)
        processed_cad_file_refined = os.path.join(processed_cad_data_folder, "processed_mesh_refined_%d.obj"%id)
        status = read_process_and_save_geometry(cad_file, processed_cad_file, processed_cad_file_refined,
                                                with_random_rotation=with_random_rotation,
                                                with_scaling_to_unit_box=with_scaling_to_unit_box,
                                                mesh_refine_size=mesh_refine_size,
                                                show=show_refined_mesh)
        if status:
            id += 1


def prepare_volume_mesh_file(n_objects,
                             processed_cad_data_folder,
                             volume_mesh_data_folder,
                             n_volume_points=5000,
                             delete_old=True,
                             use_refined_mesh_for_sdf=False):
    if delete_old:
        shutil.rmtree(volume_mesh_data_folder, ignore_errors=True)

    if not os.path.isdir(volume_mesh_data_folder):
        os.makedirs(volume_mesh_data_folder)

    if n_volume_points is None:
        n_volume_points = np.random.randint(1000, 10000)

    id = 0
    refined_cad_files_list = [x for x in os.listdir(processed_cad_data_folder) if '_refined' in x]
    cad_files_list = [x.replace('_refined', '') for x in refined_cad_files_list]
    assert len(cad_files_list) == len(refined_cad_files_list)

    idxs = np.random.randint(0, len(cad_files_list), n_objects)
    for idx in tqdm.tqdm(idxs):
        cad_file = cad_files_list[idx]
        refined_cad_file = refined_cad_files_list[idx]
        cad_file = os.path.join(processed_cad_data_folder, cad_file)
        refined_cad_file = os.path.join(processed_cad_data_folder, refined_cad_file)
        mesh = trimesh.load(cad_file, force='mesh', skip_materials=True)
        refined_mesh = trimesh.load(refined_cad_file, force='mesh', skip_materials=True)
        surface_points = refined_mesh.vertices
        surface_faces  = refined_mesh.faces
        volume_points = get_volume_points_randomly(n_volume_points)
        if use_refined_mesh_for_sdf:  # slower and unnecessary unless the surfaces are re-meshed
            volume_sdfs = get_sdf_points(refined_mesh, volume_points)
        else:
            volume_sdfs = get_sdf_points(mesh, volume_points)
        vtk_out_file = os.path.join(volume_mesh_data_folder, "volume_mesh_%d.vtk" % id)
        write_to_file(surface_points, volume_points, volume_sdfs, surface_faces, vtk_out_file)
        id += 1


def get_sdf_data_list(rnd_idx, data_folder, edge_method='edge', edge_params=None,
                      no_global=False, no_edge=False, with_normals=False, with_sdf_signs=True,
                      reversed_edge_already_included=False, self_edge_already_included=False):
    graph_data_list = []
    for i in tqdm.tqdm(rnd_idx):
        mesh_file = os.path.join(data_folder, "volume_mesh_%d.vtk" % i)
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
    return graph_data_list


def get_sdf_data_loader_3d(n_objects, processed_cad_data_folder, volume_mesh_data_folder=None,
                           batch_size=1, eval_frac=0.2, i_start=0, reversed_edge_already_included=False,
                           n_volume_points=5000,edge_method='edge', edge_params=None, no_global=False,
                           self_edge_already_included=False, with_sdf_signs=True, data_parallel=False, shuffle=False,
                           no_edge=False, with_normals=False, random_seed=None):

    if volume_mesh_data_folder is None:
        parent_folder, folder_name = os.path.split(processed_cad_data_folder)
        processed_folder_name = folder_name + "_with_volume_mesh"
        volume_mesh_data_folder = os.path.join(parent_folder, processed_folder_name)

    prepare_volume_mesh_file(n_objects, processed_cad_data_folder, volume_mesh_data_folder,
                             n_volume_points=n_volume_points)

    # random splitting into train and test
    if random_seed is not None:
        np.random.seed(random_seed)
    random_idx = np.random.permutation(range(i_start, n_objects))
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]
    test_idx = random_idx[int((1 - eval_frac) * n_objects):]

    train_graph_data_list = get_sdf_data_list(train_idx, volume_mesh_data_folder, edge_method=edge_method,
                                              edge_params=edge_params,
                                              with_sdf_signs=with_sdf_signs,
                                              no_global=no_global, no_edge=no_edge, with_normals=with_normals,
                                              reversed_edge_already_included=reversed_edge_already_included,
                                              self_edge_already_included=self_edge_already_included)

    test_graph_data_list  = get_sdf_data_list(test_idx, volume_mesh_data_folder, edge_method=edge_method, edge_params=edge_params,
                                              no_global=no_global, no_edge=no_edge, with_normals=with_normals,
                                              with_sdf_signs=with_sdf_signs,
                                              reversed_edge_already_included=reversed_edge_already_included,
                                              self_edge_already_included=self_edge_already_included)

    if data_parallel:
        train_data = DataListLoader(train_graph_data_list, batch_size=batch_size, shuffle=shuffle)
        test_data = DataListLoader(test_graph_data_list, batch_size=batch_size, shuffle=shuffle)
    else:
        train_data = DataLoader(train_graph_data_list, batch_size=batch_size, shuffle=shuffle)
        test_data = DataLoader(test_graph_data_list, batch_size=batch_size, shuffle=shuffle)
    return train_data, test_data




if __name__ == '__main__':
    n_objects = 50
    data_format = "/*.obj"
    cad_data_folder = r"D:\data\space_claim_round2\all"
    # data_format = "/**/geomFiles/geom.obj"
    # cad_data_folder = r"D:\data\space_claim_round3\data"
    processed_cad_data_folder = r"D:\data\space_claim_round2\data_processed"
    # prepare_processed_cad_data_folder(cad_data_folder,
    #                                   processed_cad_data_folder=processed_cad_data_folder,
    #                                   data_format=data_format,
    #                                   with_random_rotation=True,
    #                                   with_scaling_to_unit_box=True,
    #                                   mesh_refine_size=0.1,
    #                                   show_refined_mesh=False)

    get_sdf_data_loader_3d(n_objects, processed_cad_data_folder,
                           edge_method='proximity', edge_params={'radius': 0.1},
                           with_sdf_signs=False)