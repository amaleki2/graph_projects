import numpy as np
import pulp
from scipy.spatial import cKDTree as KDTree
import trimesh
from skimage.measure import marching_cubes
from scipy.spatial import distance_matrix
from scipy.optimize import linprog


def generate_surface_mesh_from_sdf(sdf_voxel, levelset=0.0):
    voxels_res = sdf_voxel.shape[0]
    verts, faces, normals, values = marching_cubes(sdf_voxel, level=levelset, spacing=[1/voxels_res] * 3)
    trimesh_mesh = trimesh.Trimesh(verts, faces)
    return trimesh_mesh


def compute_chamfer_distance(gt_sdf_voxel, pr_sdf_voxel, num_mesh_samples=30000, levelset=0.0):
    gt_mesh = generate_surface_mesh_from_sdf(gt_sdf_voxel, levelset=levelset)
    pr_mesh = generate_surface_mesh_from_sdf(pr_sdf_voxel, levelset=levelset)

    gt_mesh_sampled_points = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
    pr_mesh_sampled_points = trimesh.sample.sample_surface(pr_mesh, num_mesh_samples)[0]

    gt_points_kd_tree = KDTree(gt_mesh_sampled_points)
    d1, _ = gt_points_kd_tree.query(pr_mesh_sampled_points)
    gt_to_gen_chamfer = np.mean(np.square(d1))

    pr_points_kd_tree = KDTree(pr_mesh_sampled_points)
    d2, _ = pr_points_kd_tree.query(gt_mesh_sampled_points)
    gen_to_gt_chamfer = np.mean(np.square(d2))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def compute_wmd_scipy(s1, s2):
    n = len(s1)
    d = distance_matrix(s1, s2)
    A = np.zeros((2 * n, n * n))
    b = np.ones(2 * n)
    for i in range(n):
        A[i, i*n:(i+1)*n] = 1
    for i in range(n):
        A[i + n, i::n] = 1

    c = d.reshape(-1)
    A_eq = A
    b_eq = b
    bounds = (0, 1)
    sol = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    assignments = sol.x.reshape(n, n)
    loss = sol.fun
    return loss, assignments


def compute_wmd_pulp(s1, s2, integer_tag_on=True):
    n = len(s1)
    d = distance_matrix(s1, s2)
    prob = pulp.LpProblem("world mover distance", pulp.LpMinimize)
    if integer_tag_on:
        vars = [pulp.LpVariable("x%d"%i, 0, 1, pulp.LpInteger) for i in range(n * n)]
    else:
        vars = [pulp.LpVariable("x%d"%i, 0, 1) for i in range(n * n)]

    eq = 0
    for i in range(n):
        for j in range(n):
            eq += d[i, j] * vars[i * n + j]
    prob += eq

    for i in range(n):
        ieq = 0
        for j in range(n):
            ieq += vars[i * n + j]
        prob += ieq == 1

    for i in range(n):
        ieq = 0
        for j in range(n):
            ieq += vars[i + j * n]
        prob += ieq == 1
    prob.solve()
    loss = pulp.value(prob.objective) / n
    status = pulp.LpStatus[prob.status]
    return loss, status


def compute_world_mover_distance(gt_sdf_voxel, pr_sdf_voxel, num_mesh_samples=500, levelset=0.0):
    gt_mesh = generate_surface_mesh_from_sdf(gt_sdf_voxel, levelset=levelset)
    pr_mesh = generate_surface_mesh_from_sdf(pr_sdf_voxel, levelset=levelset)

    gt_mesh_sampled_points = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
    pr_mesh_sampled_points = trimesh.sample.sample_surface(pr_mesh, num_mesh_samples)[0]

    loss, status = compute_wmd_pulp(gt_mesh_sampled_points, pr_mesh_sampled_points)
    return loss


def compute_mesh_accuracy(gt_sdf_voxel, pr_sdf_voxel, levelset=0.0, num_mesh_samples=30000):
    n_90percent = int(0.9 * num_mesh_samples)
    gt_mesh = generate_surface_mesh_from_sdf(gt_sdf_voxel, levelset=levelset)
    pr_mesh = generate_surface_mesh_from_sdf(pr_sdf_voxel, levelset=levelset)

    gt_mesh_sampled_points = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
    pr_mesh_sampled_points = trimesh.sample.sample_surface(pr_mesh, num_mesh_samples)[0]

    gt_points_kd_tree = KDTree(gt_mesh_sampled_points)
    d1, _ = gt_points_kd_tree.query(pr_mesh_sampled_points)
    d1.sort()
    return d1[n_90percent]


def repeat_metric(gt_sdf_voxel, pr_sdf_voxel, metric, repeats=10, **kwargs):
    losses = []
    for _ in range(repeats):
        out = metric(gt_sdf_voxel, pr_sdf_voxel, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        losses.append(out)
    return losses



if __name__ == '__main__':
    grid_x, grid_y, grid_z = np.mgrid[-1:1:128j, -1:1:128j, -1:1:128j]
    sdf1 = np.sqrt(grid_x ** 2 + grid_y ** 2 + grid_z ** 2) - 0.5  # sphere of radius 0.5
    grid_x, grid_y, grid_z = np.mgrid[-1:1:32j, -1:1:32j, -1:1:32j]
    sdf2 = np.sqrt(grid_x ** 2 + grid_y ** 2 + grid_z ** 2) - 0.5  # sphere of radius 0.5
    cd = compute_chamfer_distance(sdf1, sdf2)
    # wmd = compute_world_mover_distance(sdf1, sdf2)
    ma = compute_mesh_accuracy(sdf1, sdf2)
    # a = np.random.random((100, 2))
    # b = np.random.random((100, 2))
    # loss, status = compute_wmd2(a, b)
    # print(loss, status)
    # idx = np.argmax(sol.x.reshape(len(a), len(a)), axis=1)
    # import matplotlib.pyplot as plt
    # plt.scatter(a[:, 0], a[:, 1], c='r', s=40, marker='o')
    # plt.scatter(b[:, 0], b[:, 1], c='b', s=40, marker='s')
    # for i in range(len(a)):
    #     plt.plot([a[i, 0], b[idx[i], 0]], [a[i, 1], b[idx[i], 1]], 'k--')
    # print(sol.fun)
    # plt.show()