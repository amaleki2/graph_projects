import torch
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot_sdf_results(model, data_loader, save_name="", max_num_data=10, output_func=lambda x: x, levels=None, unit_square=True):
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
            if i == max_num_data: break
            points = data.x.numpy()
            xx = points[:, 0]
            yy = points[:, 1]
            true_vals = data.y.numpy()[:, 0]

            data = data.to(device=device)
            output = model(data)
            pred_vals = output_func(output)

            plot_scatter_contour(xx, yy, true_vals, pred_vals, levels=levels, unit_square=unit_square)
            plt.show()


def plot_scatter_contour(xx, yy, true_vals, pred_vals, levels=None, unit_square=True):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 5), nrows=1, ncols=3)

    cntr1 = ax1.tricontour(xx, yy, true_vals, levels=levels, linewidths=1, colors='k')
    plt.clabel(cntr1, fmt='%0.2f', colors='k', fontsize=10)
    cntr1 = ax1.tricontourf(xx, yy, true_vals, cmap="RdBu_r", levels=20)
    fig.colorbar(cntr1, ax=ax1)
    if unit_square: ax1.set(xlim=(-1, 1), ylim=(-1, 1))
    ax1.set_xticks([]);
    ax1.set_yticks([])

    cntr2 = ax2.tricontour(xx, yy, pred_vals, levels=levels, linewidths=1, colors='k')
    plt.clabel(cntr2, fmt='%0.2f', colors='k', fontsize=10)
    cntr2 = ax2.tricontourf(xx, yy, pred_vals, cmap="RdBu_r", levels=20)
    fig.colorbar(cntr2, ax=ax2)
    if unit_square: ax2.set(xlim=(-1, 1), ylim=(-1, 1))
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
    if unit_square: ax3.set(xlim=(-1, 1), ylim=(-1, 1))
    ax3.set_xticks([]);
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
