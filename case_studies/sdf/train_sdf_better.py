import torch
import numpy as np
from src import train_forward_step
from case_studies.sdf.train_sdf import print_and_save
from case_studies.sdf.get_data_sdf_new import get_sdf_data_loader_3d


def train_sdf_with_shuffling(model,
                             processed_cad_data_folder,
                             loss_funcs,
                             update_data_every=5,
                             n_objects=1000,
                             eval_frac=0.2,
                             n_epochs=500,
                             edge_method='edge',
                             edge_params=None,
                             no_global=False,
                             no_edge=False,
                             print_every=25,
                             with_normals=False,
                             include_reverse_edges=False,
                             device=torch.device('cpu'),
                             save_name="",
                             lr_0=0.001,
                             lr_scheduler_step_size=50,
                             lr_scheduler_gamma=0.5,
                             batch_size=1,
                             n_volume_points=5000,
                             shuffle=False,
                             n_jobs=1,
                             **losses_params):

    data_parallel = isinstance(device, list)
    if data_parallel:  # data parallel
        device0 = torch.device('cuda:%d'%device[0])
        model = model.to(device0)
    else:
        model = model.to(device=device)

    i_start = 0
    train_losses_list = []
    test_losses_list = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

    random_idx = np.random.permutation(range(n_objects))
    train_idx = random_idx[:int((1 - eval_frac) * n_objects)]

    for epoch in range(i_start, n_epochs + 1):
        if epoch % update_data_every == 0:
            print("computing volume mesh files ...")
            train_data_loader, test_data_loader = \
                get_sdf_data_loader_3d(n_objects, processed_cad_data_folder,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       n_volume_points=n_volume_points,
                                       edge_method=edge_method,
                                       edge_params=edge_params,
                                       no_edge=no_edge,
                                       no_global=no_global,
                                       with_normals=with_normals,
                                       data_parallel=data_parallel,
                                       include_reverse_edges=include_reverse_edges,
                                       train_idx=train_idx,
                                       n_jobs=n_jobs)

        epoch_loss = []
        for data in train_data_loader:
            optimizer.zero_grad()
            losses = train_forward_step(model, data, loss_funcs, device, **losses_params)
            epoch_loss.append([ll.item() for ll in losses])
            train_loss = sum(losses)
            train_loss.backward()
            optimizer.step()
        epoch_loss_mean = np.mean(epoch_loss, axis=0)
        train_losses_list.append(epoch_loss_mean)

        if epoch % print_every == 0:
            test_epoch_loss = []

            for data in test_data_loader:
                losses = train_forward_step(model, data, loss_funcs, device, training=False)
                test_epoch_loss.append([ll.item() for ll in losses])
            test_epoch_loss_mean = np.mean(test_epoch_loss, axis=0)
            test_losses_list.append(test_epoch_loss_mean)
            print_and_save(model, epoch, train_losses_list, test_losses_list, optimizer, save_name)

        scheduler.step()





