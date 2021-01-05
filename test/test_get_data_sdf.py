from src import get_sdf_data_loader

data_folder = ""
batch_size = 1
data, _ = get_sdf_data_loader(1, data_folder, batch_size, eval_frac=0.0, edge_method='edge', edge_params=None)
data = next(iter(data))
cells_sets = [set(x) for x in data.face.T.tolist()]
assert len(cells_sets) == 14
assert {3, 6, 11} in cells_sets
assert {2, 6, 10} in cells_sets
assert {3, 7, 11} in cells_sets
assert {6, 10, 11} in cells_sets
assert {9, 7, 11} in cells_sets
assert {9, 10, 11} in cells_sets
assert {10, 9, 8} in cells_sets
assert {10, 8, 5} in cells_sets
assert {10, 2, 5} in cells_sets
assert {5, 8, 1} in cells_sets
assert {8, 1, 4} in cells_sets
assert {8, 9, 4} in cells_sets
assert {0, 9, 4} in cells_sets
assert {7, 9, 0} in cells_sets


edges_set = [set(x) for x in data.edge_index.T.tolist()]
assert len(edges_set) == 25 * 2 + 12
for i in range(12):
    assert {i} in edges_set

assert {0, 4} in edges_set
assert {0, 7} in edges_set
assert {0, 9} in edges_set
assert {1, 4} in edges_set
assert {1, 5} in edges_set
assert {1, 8} in edges_set
assert {2, 5} in edges_set
assert {2, 6} in edges_set
assert {2, 10} in edges_set
assert {3, 6} in edges_set
assert {3, 7} in edges_set
assert {3, 11} in edges_set
assert {4, 8} in edges_set
assert {4, 9} in edges_set
assert {5, 8} in edges_set
assert {5, 10} in edges_set
assert {6, 10} in edges_set
assert {6, 11} in edges_set
assert {7, 9} in edges_set
assert {7, 11} in edges_set
assert {8, 9} in edges_set
assert {8, 10} in edges_set
assert {9, 10} in edges_set
assert {9, 11} in edges_set
assert {10, 11} in edges_set


data, _ = get_sdf_data_loader(1, data_folder, batch_size, eval_frac=0.0, edge_method='proximity',
                              edge_params={'radius': 1.5})
data = next(iter(data))
edges_set = [set(x) for x in data.edge_index.T.tolist()]
assert len(edges_set) == 38 * 2 + 12
assert {0, 4} in edges_set
assert {0, 7} in edges_set
assert {0, 9} in edges_set
assert {1, 4} in edges_set
assert {1, 5} in edges_set
assert {1, 8} in edges_set
assert {2, 5} in edges_set
assert {2, 6} in edges_set
assert {2, 10} in edges_set
assert {3, 6} in edges_set
assert {3, 7} in edges_set
assert {3, 11} in edges_set
assert {4, 8} in edges_set
assert {4, 9} in edges_set
assert {5, 8} in edges_set
assert {5, 10} in edges_set
assert {6, 10} in edges_set
assert {6, 11} in edges_set
assert {7, 9} in edges_set
assert {7, 11} in edges_set
assert {8, 9} in edges_set
assert {8, 10} in edges_set
assert {9, 10} in edges_set
assert {9, 11} in edges_set
assert {10, 11} in edges_set
assert {1, 9} in edges_set
assert {1, 10} in edges_set
assert {3, 9} in edges_set
assert {3, 10} in edges_set
assert {4, 5} in edges_set
assert {4, 7} in edges_set
assert {4, 10} in edges_set
assert {5, 6} in edges_set
assert {5, 9} in edges_set
assert {6, 7} in edges_set
assert {6, 9} in edges_set
assert {7, 10} in edges_set
assert {8, 11} in edges_set

for i in range(12):
    assert {i} in edges_set

edge_params = {'radius': 1.5}
data, _ = get_sdf_data_loader(1, data_folder, batch_size, eval_frac=0.0, edge_method='both',
                              edge_params=edge_params)
data = next(iter(data))
edges_set = [set(x) for x in data.edge_index.T.tolist()]
assert len(edges_set) == 38 * 2 + 25 * 2 + 12
assert {0, 4} in edges_set
assert {0, 7} in edges_set
assert {0, 9} in edges_set
assert {1, 4} in edges_set
assert {1, 5} in edges_set
assert {1, 8} in edges_set
assert {2, 5} in edges_set
assert {2, 6} in edges_set
assert {2, 10} in edges_set
assert {3, 6} in edges_set
assert {3, 7} in edges_set
assert {3, 11} in edges_set
assert {4, 8} in edges_set
assert {4, 9} in edges_set
assert {5, 8} in edges_set
assert {5, 10} in edges_set
assert {6, 10} in edges_set
assert {6, 11} in edges_set
assert {7, 9} in edges_set
assert {7, 11} in edges_set
assert {8, 9} in edges_set
assert {8, 10} in edges_set
assert {9, 10} in edges_set
assert {9, 11} in edges_set
assert {10, 11} in edges_set
assert {1, 9} in edges_set
assert {1, 10} in edges_set
assert {3, 9} in edges_set
assert {3, 10} in edges_set
assert {4, 5} in edges_set
assert {4, 7} in edges_set
assert {4, 10} in edges_set
assert {5, 6} in edges_set
assert {5, 9} in edges_set
assert {6, 7} in edges_set
assert {6, 9} in edges_set
assert {7, 10} in edges_set
assert {8, 11} in edges_set

for i in range(12):
    assert {i} in edges_set

assert edge_params['e1'] == 25
print("tests passed successfully!")
