import torch
import numpy as np
import torch.nn as nn

from torch_scatter import scatter_mean


# all loss functions defined below have a unused kwargs so that a single signature can be used for all of them
def regular_loss(pred, data, loss_func=nn.L1Loss, **kwargs):
    loss = loss_func()(pred, data.y)
    return loss


def borderless_loss(pred, data, loss_func=nn.L1Loss, radius=0.1, **kwargs):
    mask = torch.logical_and(torch.abs(data.x[:, 0]) < 1 - radius,
                             torch.abs(data.x[:, 1]) < 1 - radius)
    loss = loss_func(reduction='none')(pred, data.y)
    loss_masked = loss[mask]
    loss_masked_reduced = torch.mean(loss_masked)
    return loss_masked_reduced


def inner_shale_loss(pred, data, loss_func=nn.L1Loss, **kwargs):
    mask = torch.abs(data.y < 0)
    loss = loss_func(reduction='none')(pred, data.y)
    loss_inner = loss[mask]
    loss_inner_reduced = torch.mean(loss_inner)
    return loss_inner_reduced


def boundary_loss(pred, data, loss_func=nn.L1Loss, **kwargs):
    mask = torch.abs(data.y) < 0.1
    loss = loss_func(reduction='none')(pred, data.y)
    loss_boundary = loss[mask]
    loss_boundary_reduced = torch.mean(loss_boundary)
    return loss_boundary_reduced


def sign_loss(pred, data, loss_func=nn.L1Loss, **kwargs):
    mask = pred * data.y < 0.
    loss = loss_func(reduction='none')(pred, data.y)
    loss_sign = loss[mask]
    loss_sign_reduced = torch.mean(loss_sign)
    return loss_sign_reduced


def clamped_loss(pred, data, loss_func=nn.L1Loss, maxv=0.05, **kwargs):
    loss = loss_func(reduction='none')(pred, data.y)
    loss_clamped = torch.clamp(loss, max=maxv)
    loss_clamped_reduced = torch.mean(loss_clamped)
    return loss_clamped_reduced


def corner_loss(pred, data, loss_func=nn.L1Loss):
    points = data.x[:, :2]
    img = data.x[:, 2]
    center = torch.mean(points[img == 1], dim=0)
    dist_from_center = torch.sum(torch.square(points - center), dim=1) - 1000 * (img == 0)
    mask = dist_from_center > torch.sort(dist_from_center)[0][-5]
    loss = loss_func(reduction='none')(pred, data.y)
    loss_corner = loss[mask]
    loss_corner_reduced = torch.mean(loss_corner)
    return loss_corner_reduced


def graph_loss(pred, data, loss_func=nn.L1Loss, aggr_func=None, **kwargs):
    if aggr_func is None:
        aggr_func = lambda x: sum(x) / len(x)

    if isinstance(pred, list):
        loss = [loss_func()(out[1], data.y) for out in pred]
        loss = aggr_func(loss)
    else:
        loss = loss_func()(pred[1], data.y)
    return loss


def graph_loss_data_parallel(pred, data, loss_func=nn.L1Loss, aggr_func=None, **kwargs):
    if aggr_func is None:
        aggr_func = lambda x: sum(x) / len(x)

    device = pred[0][1].device
    data_y = torch.cat([d.y for d in data]).to(device)
    if isinstance(pred, list):
        loss = [loss_func()(out[1], data_y) for out in pred]
        loss = aggr_func(loss)
    else:
        loss = loss_func()(pred[1], data_y)
    return loss


def graph_loss_data_parallel_zero_focused(pred, data, loss_func=nn.L1Loss, aggr_func=None, **kwargs):
    if aggr_func is None:
        aggr_func = lambda x: sum(x) / len(x)

    device = pred[0][1].device
    data_y = torch.cat([d.y for d in data]).to(device)
    if isinstance(pred, list):
        loss = [loss_func()(out[1], data_y) for out in pred]
        loss = aggr_func(loss)
    else:
        mask = abs(data_y) < 0.1
        loss = loss_func()(pred[1][mask], data_y[mask])
    return loss


def clamped_loss_data_parallel(pred, data, loss_func=nn.L1Loss, maxv=0.1, **kwargs):
    device = pred[0][1].device
    data_y = torch.cat([d.y for d in data]).to(device)
    loss = loss_func(reduction='none')(pred[1], data_y)
    loss_clamped = torch.clamp(loss, max=maxv)
    loss_clamped_reduced = torch.mean(loss_clamped)
    return loss_clamped_reduced

# def graph_loss_data_parallel_batched(pred, data, loss_func=nn.L1Loss, aggr_func=None, **kwargs):
#     if aggr_func is None:
#         aggr_func = lambda x: sum(x) / len(x)
#
#     data_y = torch.cat([d.y for d in data])
#     if isinstance(pred, list):
#         raise NotImplementedError
#     else:
#         lens = [len(d) for d in data]
#         splits = np.cumsum(lens)
#         outs = torch.split(pred[1], splits)
#         loss = [loss_func()(o, d.y) for (o, d) in zip(outs, data)]
#         loss = aggr_func(loss)
#     return loss


def graph_loss_batched(pred, data, loss_func=nn.L1Loss, **kwargs):
    aggr_func = lambda x: scatter_mean(x, index=data.batch, dim=0)
    return graph_loss(pred, data, loss_func=loss_func, aggr_func=aggr_func, **kwargs)


def banded_loss(xpred, xtrue, loss_func=nn.L1Loss, lb=0., ub=1.):
    loss_vals = loss_func(reduction='none')(xpred, xtrue)
    mask = torch.logical_and(xtrue < ub, xtrue > lb)
    if torch.any(mask):
        loss = torch.mean(loss_vals[mask])
    else:
        loss = 0.
    return loss


def level_set_loss(pred, data, loss_func=nn.L1Loss, level_sets=[-0.1, -0.05, 0, 0.05, 0.1]):
    loss = 0
    for lb, ub in zip(level_sets[:-1], level_sets[1:]):
        loss += sum([banded_loss(out[1], data.y, loss_func=loss_func, lb=lb, ub=ub) for out in pred]) / len(pred)
    return loss

# def eplison_loss(pred, data, eps=0, **kwargs):
#     eps_loss = torch.relu(torch.abs(pred-data.y) - eps)
#     eps_loss_reduced = torch.mean(eps_loss)
#     return eps_loss_reduced

# def eikonal_loss(pred, xy, device='cuda', retain_graph=True):
#     pred.backward(gradient=torch.ones(pred.size()).to(device), retain_graph=retain_graph)
#     dg = xy.grad[:, :2]
#     dg_mag = torch.sqrt(torch.sum(dg * dg, dim=-1))
#     eikonal_loss = l2_loss()(dg_mag, torch.ones(dg_mag.size()).to(device))
#     eikonal_loss.requires_grad = True
#     return eikonal_loss
