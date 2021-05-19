import os
import torch
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pyflann
from scipy.interpolate import griddata
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


def vertex_to_proximity_kdtree(x, edge_params, n_features_to_consider=3):
    radius = edge_params['radius']
    min_n_edges, max_n_edges = edge_params['min_n_edges'], edge_params['max_n_edges']
    points = x[:, :n_features_to_consider]
    tree = KDTree(points)
    dist, idx = tree.query(points, k=max_n_edges)
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


def plot_sdf_results(model, data_loader, save_name="", max_num_data=10, output_func=lambda x: x, levels=None, plot_3d=False):
    train_loss_history = np.load("save_dir/loss_train_" + save_name + ".npy")
    test_loss_history = np.load("save_dir/loss_test_" + save_name + ".npy")
    plt.plot(train_loss_history, label="train loss")
    plt.plot(np.linspace(0, len(train_loss_history) - 1, len(test_loss_history)), test_loss_history, label="test loss")
    plt.yscale('log')
    plt.legend()

    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("save_dir/model_" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i == max_num_data:
                break
            points = data.x.numpy()[:, :3]
            true_vals = data.y.numpy()[:, 0]
            data = data.to(device=device)
            output = model(data)
            pred_vals = output_func(output)
            if plot_3d:
                plot_scatter_contour_3d(points, true_vals, pred_vals, levels=levels)
            else:
                xx = points[:, 0]
                yy = points[:, 1]
                plot_scatter_contour(xx, yy, true_vals, pred_vals, levels=levels)
            plt.show()


def plot_scatter_contour(xx, yy, true_vals, pred_vals, levels=None):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 5), nrows=1, ncols=3)

    cntr1 = ax1.tricontour(xx, yy, true_vals, levels=levels, linewidths=1, colors='k')
    plt.clabel(cntr1, fmt='%0.2f', colors='k', fontsize=10)
    cntr1 = ax1.tricontourf(xx, yy, true_vals, cmap="RdBu_r", levels=20)
    fig.colorbar(cntr1, ax=ax1)
    ax1.set(xlim=(-1, 1), ylim=(-1, 1))
    ax1.set_xticks([]);
    ax1.set_yticks([])

    cntr2 = ax2.tricontour(xx, yy, pred_vals, levels=levels, linewidths=1, colors='k')
    plt.clabel(cntr2, fmt='%0.2f', colors='k', fontsize=10)
    cntr2 = ax2.tricontourf(xx, yy, pred_vals, cmap="RdBu_r", levels=20)
    fig.colorbar(cntr2, ax=ax2)
    ax2.set(xlim=(-1, 1), ylim=(-1, 1))
    ax2.set_xticks([]);
    ax2.set_yticks([])
    #     if levels:
    #         new_levels = [(l + r) / 2 for (l, r) in zip(levels[1:], levels[:-1])] + levels
    #         new_levels = sorted(new_levels)
    #     else:
    #         new_levels = None
    new_levels = levels
    cntr3 = ax3.tricontour(xx, yy, true_vals, levels=new_levels, linewidths=2, colors='k')
    plt.clabel(cntr3, fmt='%0.2f', colors='k', fontsize=10)
    cntr3 = ax3.tricontour(xx, yy, pred_vals, levels=new_levels, linewidths=1, colors='r', linestyles='--')
    ax3.set(xlim=(-1, 1), ylim=(-1, 1))
    ax3.set_xticks([]);
    ax3.set_yticks([])
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def plot_scatter_contour_3d(points, true_vals, pred_vals, levels=None):
    ends, n_pts = -0.9, 100
    n_pnts_c = n_pts * 1j
    x = np.linspace(-ends, ends, n_pts, endpoint=True)
    X, Y, Z = np.mgrid[-ends:ends:n_pnts_c, -ends:ends:n_pnts_c, -ends:ends:n_pnts_c]
    SDFS_true = griddata(points, true_vals, (X, Y, Z))
    SDFS_pred = griddata(points, pred_vals, (X, Y, Z))
    fig, axes = plt.subplots(figsize=(20, 20), nrows=4, ncols=3)
    for i in range(4):
        ax1, ax2, ax3 = axes[i]
        z_slice = 20 * i + 5
        cntr1 = ax1.contour(x, x, SDFS_true[:, :, z_slice], levels=levels, linewidths=1, colors='k')
        plt.clabel(cntr1, fmt='%0.2f', colors='k', fontsize=10)
        cntr1 = ax1.contourf(x, x, SDFS_true[:, :, z_slice], cmap="RdBu_r", levels=20)
        fig.colorbar(cntr1, ax=ax1)
        ax1.set(xlim=(-1, 1), ylim=(-1, 1))
        ax1.set_xticks([])
        ax1.set_yticks([])

        cntr2 = ax2.contour(x, x, SDFS_pred[:, :, z_slice], levels=levels, linewidths=1, colors='k')
        plt.clabel(cntr2, fmt='%0.2f', colors='k', fontsize=10)
        cntr2 = ax2.contourf(x, x, SDFS_pred[:, :, z_slice], cmap="RdBu_r", levels=20)
        fig.colorbar(cntr2, ax=ax2)
        ax2.set(xlim=(-1, 1), ylim=(-1, 1))
        ax2.set_xticks([])
        ax2.set_yticks([])
        #     if levels:
        #         new_levels = [(l + r) / 2 for (l, r) in zip(levels[1:], levels[:-1])] + levels
        #         new_levels = sorted(new_levels)
        #     else:
        #         new_levels = None
        new_levels = levels
        cntr3 = ax3.contour(x, x, SDFS_true[:, :, z_slice], levels=new_levels, linewidths=2, colors='k')
        plt.clabel(cntr3, fmt='%0.2f', colors='k', fontsize=10)
        cntr3 = ax3.contour(x, x, SDFS_pred[:, :, z_slice], levels=new_levels, linewidths=1, colors='r', linestyles='--')
        ax3.set(xlim=(-1, 1), ylim=(-1, 1))
        ax3.set_xticks([])
        ax3.set_yticks([])
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def plot_sdf_results_over_line(model, data, lines=(-0.5, 0, 0.5), save_name="", max_num_data=10):
    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load("save_dir/model_" + save_name + ".pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(data):
            if i == max_num_data: break
            d = d.to(device=device)
            pred = model(d)
            if isinstance(pred, list):
                pred = pred[-1]
            pred = pred[1]
            pred = pred.numpy()[:, 0]
            gt = d.y.numpy()[:, 0]

            cells = d.face.numpy()
            points = d.x.numpy()
            points[:, 2] = 0.
            mesh = meshio.Mesh(points=points, cells=[("triangle", cells.T)])

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            for line in lines:
                plot_mesh_onto_line(mesh, val=pred, x=line)
                plot_mesh_onto_line(mesh, val=gt, x=line, linestyle="--")

            plt.subplot(1, 2, 2)
            for line in lines:
                plot_mesh_onto_line(mesh, val=pred, y=line)
                plot_mesh_onto_line(mesh, val=gt, y=line, linestyle="--")
            plt.subplots_adjust(wspace=0.2)
            plt.show()


def plot_mesh_onto_line(mesh, val, x=None, y=None, show=False, linestyle="-"):
    if not isinstance(mesh.points, np.ndarray):
        mesh.points = np.array(mesh.points)
    assert (x is None) ^ (y is None), "one of x or y has to be None"
    if x is None:
        x = np.linspace(-1, 1, 50)
        y = np.ones_like(x) * y
        plotting_axes = x
    else:  # y is None:
        y = np.linspace(-1, 1, 50)
        x = np.ones_like(y) * x
        plotting_axes = y

    nodes_x = mesh.points[:, 0]
    nodes_y = mesh.points[:, 1]
    elements_tris = [c for c in mesh.cells if c.type == "triangle"][0].data
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_tris)
    interpolator = tri.LinearTriInterpolator(triangulation, val)
    val_over_line = interpolator(x, y)
    plt.plot(plotting_axes, val_over_line, linestyle=linestyle)
    if show: plt.show()


def plot_mesh(mesh, dims=2, node_labels=False, vals=None, with_colorbar=False, levels=None, border=None,
              ticks_off=True):
    if not isinstance(mesh.points, np.ndarray):
        mesh.points = np.array(mesh.points)
    nodes_x = mesh.points[:, 0]
    nodes_y = mesh.points[:, 1]
    if dims == 2:
        elements_tris = [c for c in mesh.cells if c.type == "triangle"][0].data
        # plt.figure(figsize=(8, 8))
        if vals is None:
            plt.triplot(nodes_x, nodes_y, elements_tris, alpha=0.9, color='r')
        else:
            triangulation = tri.Triangulation(nodes_x, nodes_y, elements_tris)
            p = plt.tricontourf(triangulation, vals, 30)
            if with_colorbar:
                plt.colorbar()
            if levels:
                cn = plt.tricontour(triangulation, vals, levels, colors='w')
                plt.clabel(cn, fmt='%0.2f', colors='k', fontsize=10)
        if border:
            plt.hlines(1 - border, -1 + border, 1 - border, 'r')
            plt.hlines(-1 + border, -1 + border, 1 - border, 'r')
            plt.vlines(1 - border, -1 + border, 1 - border, 'r')
            plt.vlines(-1 + border, -1 + border, 1 - border, 'r')
        if ticks_off:
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

    if node_labels:
        for i, (x, y) in enumerate(zip(nodes_x, nodes_y)):
            plt.text(x, y, i)

    if vals is not None:
        return p
