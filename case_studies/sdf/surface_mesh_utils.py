import gmsh
import sys
import meshio
import numpy as np
import tqdm
import trimesh


def generate_gmsh_edge_and_cells(cells):
    # as per convenction of gmsh, if an edge is used in the reverse order,
    # its index in the cell is negative. but  because -0 = 0,
    # I can't have any edge at index 0, so fill the first element with None
    edges = [None]
    gmsh_cells = []
    idx = 1
    for cell in tqdm.tqdm(cells):
        gmsh_cell = []
        if [cell[0], cell[1]] not in edges:
            if [cell[1], cell[0]] not in edges:
                edges.append([cell[0], cell[1]])
                gmsh_cell.append(idx)
                idx += 1
            else:
                gmsh_cell.append(-edges.index([cell[1], cell[0]]))
        else:
            gmsh_cell.append(edges.index([cell[0], cell[1]]))

        if [cell[1], cell[2]] not in edges:
            if [cell[2], cell[1]] not in edges:
                edges.append([cell[1], cell[2]])
                gmsh_cell.append(idx)
                idx += 1
            else:
                gmsh_cell.append(-edges.index([cell[2], cell[1]]))
        else:
            gmsh_cell.append(edges.index([cell[1], cell[2]]))

        if [cell[2], cell[0]] not in edges:
            if [cell[0], cell[2]] not in edges:
                edges.append([cell[2], cell[0]])
                gmsh_cell.append(idx)
                idx += 1
            else:
                gmsh_cell.append(-edges.index([cell[0], cell[2]]))
        else:
            gmsh_cell.append(edges.index([cell[2], cell[0]]))

        gmsh_cells.append(gmsh_cell)
    edges = np.array(edges)
    gmsh_cells = np.array(gmsh_cells)
    return edges, gmsh_cells


def to_gmsh(points, edges, cells, lc=0.1):
    point_dict = {}
    last_point_idx = gmsh.model.geo.getMaxTag(0)
    for ip, p in enumerate(points):
        gmsh.model.geo.addPoint(p[0], p[1], p[2], lc, 1 + last_point_idx + ip)
        point_dict[ip] = 1 + last_point_idx + ip

    edge_dict = {}
    last_edge_idx = gmsh.model.geo.getMaxTag(1)
    for ie, e in enumerate(edges):
        if e is None: continue
        p1 = point_dict[e[0]]
        p2 = point_dict[e[1]]
        gmsh.model.geo.addLine(p1, p2, 1 + last_edge_idx + ie)
        edge_dict[ie] = 1 + last_edge_idx + ie

    last_cell_idx = gmsh.model.geo.getMaxTag(2)
    for ic, cell in enumerate(cells):
        edge_list = [edge_dict[edge] if edge >= 0 else -edge_dict[-edge] for edge in cell]
        gmsh.model.geo.addCurveLoop(edge_list, 1 + ic + last_cell_idx)
        gmsh.model.geo.addPlaneSurface([1 + ic + last_cell_idx], 1 + ic + last_cell_idx)

    last_volume_idx = gmsh.model.geo.getMaxTag(3)
    last_cell_idx_now = gmsh.model.geo.getMaxTag(2)
    l = list(range(last_cell_idx + 1, last_cell_idx_now + 1))
    gmsh.model.geo.addSurfaceLoop(l, last_cell_idx_now + 1)
    gmsh.model.geo.addVolume([last_cell_idx_now + 1], 1 + last_volume_idx)


def test():
    gmsh.initialize(sys.argv)
    gmsh.model.add("test")
    points = np.array(
              [[0, 0, 0],
               [1, 0, 0],
               [1, 1, 0],
               [0, 1, 0],
               [0.5, 0.5, 1]])

    cells = np.array(
             [[0, 1, 2],
              [2, 3, 0],
              [0, 1, 4],
              [1, 2, 4],
              [2, 3, 4],
              [3, 0, 4]])

    cells = np.sort(cells, axis=1)
    edges, gmsh_cells = generate_gmsh_edge_and_cells(cells)

    to_gmsh(points, edges, gmsh_cells)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.fltk.run()
    gmsh.finalize()


def scale_to_unit_box(mesh):
    points = mesh.points
    m1, m2 = points.min(), points.max()
    points = (points - m1) / (m2 - m1) * 2 - 1
    mesh.points = points
    return mesh


def check_if_not_3d(mesh):
    return np.any(np.all(mesh.points == mesh.points[0], axis=0))


def generate_surface_mesh(mesh_file=None, mesh_points=None, mesh_faces=None, lc=0.05, with_volume_mesh=False, saved_name=None, show=False):
    if mesh_points is None or mesh_faces is None:
        mesh = meshio.read(mesh_file)
        if check_if_not_3d(mesh):
            return False

        mesh_points = mesh.points
        mesh = scale_to_unit_box(mesh)
        cells = [x for x in mesh.cells if x.type == 'triangle']
        cells = cells[0].data.astype(int)
        cells = np.sort(cells, axis=1)
        mesh_faces = cells

    gmsh_points = mesh_points
    gmsh_edges, gmsh_cells = generate_gmsh_edge_and_cells(mesh_faces)
    gmsh.initialize(sys.argv)
    gmsh.model.add("mesh")
    to_gmsh(gmsh_points, gmsh_edges, gmsh_cells, lc=lc)
    gmsh.model.geo.synchronize()
    # gmsh.model.occ.addBox(-2, -2, -2, 2, 2, 2, 1)
    # gmsh.model.occ.synchronize()
    if with_volume_mesh:
        gmsh.model.mesh.generate(3)
    else:
        gmsh.model.mesh.generate(2)
    if show:
        gmsh.fltk.run()
    if saved_name is not None:
        gmsh.write(saved_name)

    return True


def rotate(mesh, matrix=None, return_matrix=False):
    if matrix is None:
        matrix = trimesh.transformations.random_rotation_matrix()
    mesh.apply_transform(matrix)
    if return_matrix:
        return matrix


def translate(mesh, vector=None):
    if vector is None:
        vector = np.random.random(3)
    matrix = trimesh.transformations.translation_matrix(vector)
    mesh.apply_transform(matrix)


def scale(mesh, factor=None, origin=None, direction=None):
    if factor is None:
        factor = np.random.random() * 1.5 + 0.5
    matrix = trimesh.transformations.scale_matrix(factor, origin=origin, direction=direction)
    mesh.apply_transform(matrix)


def refine_surface_mesh(mesh, mesh_size=0.1, show=False):
    # gmsh does not write in obj, so have to save in vtk.
    # trimesh does not read vtk, so have to read with meshio and convert to trimesh.Trimesh
    generate_surface_mesh(mesh_points=mesh.vertices, mesh_faces=mesh.faces, lc=mesh_size,
                          saved_name='tmp.vtk', show=show)
    surface_mesh = meshio.read('tmp.vtk')
    surface_points = surface_mesh.points
    surface_faces = surface_mesh.get_cells_type('triangle')
    mesh = trimesh.Trimesh(surface_points, surface_faces)
    return mesh


