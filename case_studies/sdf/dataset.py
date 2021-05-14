import trimesh
import torch
import numpy as np
from torch_geometric.data import Batch
from trimesh.proximity import ProximityQuery

_N_VOLUME_POINTS = 5000
_MIN_X = -1
_MAX_X = 1

class SDFCollater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):
        return SDFBatch.from_mesh_list(batch, self.follow_batch)

    def __call__(self, batch):
        return self.collate(batch)


class SDFDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(SDFDataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=SDFCollater(follow_batch), **kwargs)


class SDFBatch(Batch):
    @staticmethod
    def read_and_process_mesh(obj_in_file, with_random_rotation=True):
        mesh = trimesh.load(obj_in_file, force='mesh')
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        if with_random_rotation:
            matrix = trimesh.transformations.random_rotation_matrix()
            mesh.apply_transform(matrix)

        s1 = mesh.bounding_box.centroid - (_MAX_X + _MIN_X) / 2
        s2 = (_MAX_X - _MIN_X) / np.max(mesh.bounding_box.extents)
        new_vertices = mesh.vertices - s1
        mesh.vertices = new_vertices * s2
        return mesh

    @staticmethod
    def get_sdf_points(mesh, points):
        return - ProximityQuery.signed_distance(mesh, points)

    def from_mesh_list(self, mesh_name_list, follow_batch=[]):
        data_list = []
        for mesh_name in mesh_name_list:
            mesh = self.read_and_process_mesh(mesh_name)
            surface_points, surface_edges = mesh.vertices, mesh.faces
            volume_points = np.random.random((_N_VOLUME_POINTS, 3)) * (_MAX_X - _MIN_X) + _MIN_X
            sdf_points = self.get_sdf_points(mesh, volume_points)

            x = np.concatenate((surface_points, volume_points))
            on_surface = np.zeros(len(x))
            on_surface[:len(surface_points)] = 1
            x = np.concatenate((x, on_surface), axis=1)

            y = np.concatenate((np.zeros(len(surface_points)), sdf_points))

            # do some stuff here
            data = Data(x=torch.from_numpy(x).type(torch.float32),
                        y=torch.from_numpy(y).type(torch.float32),
                        u=torch.from_numpy(u).type(torch.float32),
                        edge_index=torch.from_numpy(edges).type(torch.long),
                        edge_attr=torch.from_numpy(edge_feats).type(torch.float32))
            data_list.append(data)
        super().from_data_list(data_list, follow_batch=follow_batch)




    def

    def _get_volume_points():
        return torch.rand(_N_VOLUME_POINTS) * 2 - 1


    def _get_edges(points):
        pass

    def get_sdf_points(mesh_file):
        mesh = trimesh.load(mesh_file)
        volume_points = _get_volume_points()
        all_points = torch.cat((x, volume_points))
        volume_sdfs = _get_sdf_points(mesh, all_points)

        super().from_data_list(data_list, follow_batch=follow_batch)


import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample


if __name__ == '__main__':
    pass