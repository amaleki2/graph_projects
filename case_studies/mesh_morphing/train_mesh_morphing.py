import os
import torch
import numpy as np
from src import get_device


def train_forward_step(model, data, loss_funcs, device, training=True, **loss_kwargs):
    if training:
        model.train()
    else:
        model.eval()

    data_x, data_y = data
    data_x = data_x.to(device)
    pred_y = model(data_x)
    pred_nodes = pred_y[1]
    data_nodes = data_y.x.to(device)
    losses = [func(pred_nodes, data_nodes, **loss_kwargs) for func in loss_funcs]
    return losses


def train_mesh_morphing(model, train_data, test_data, loss_funcs, n_epochs=500, print_every=25, use_cpu=False,
                        save_name="", lr_0=0.001, lr_scheduler_step_size=50, lr_scheduler_gamma=0.5,
                        resume_training=False, **losses_params):
    device = get_device(use_cpu)
    model = model.to(device=device)
    if resume_training:
        train_losses_list = np.load("save_dir/loss_train_" + save_name + ".npy").tolist()
        test_losses_list = np.load("save_dir/loss_test_" + save_name + ".npy").tolist()
        model.load_state_dict(torch.load("save_dir/model_" + save_name + ".pth", map_location=device))
        i_start = len(train_losses_list)
        n_scheduler_activated = i_start // lr_scheduler_step_size
        lr_0 = lr_0 / (lr_scheduler_gamma ** n_scheduler_activated)
    else:
        i_start = 0
        train_losses_list = []
        test_losses_list = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

    for epoch in range(i_start, n_epochs + 1):
        epoch_loss = []
        for data in train_data:
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
            for data in test_data:
                losses = train_forward_step(model, data, loss_funcs, device, training=False)
                test_epoch_loss.append([ll.item() for ll in losses])
            test_epoch_loss_mean = np.mean(test_epoch_loss, axis=0)
            test_losses_list.append(test_epoch_loss_mean)
            print_and_save(model, epoch, train_losses_list, test_losses_list, optimizer, save_name)

        scheduler.step()


def print_and_save(model, epoch, train_losses_list, test_losses_list, optimizer, save_name):
    lr = optimizer.param_groups[0]['lr']
    print("epoch %4s: learning rate=%0.2e" %(str(epoch), lr), end="")
    print(", train loss: ", end="")
    print(['%.4f' % n for n in train_losses_list[-1]], end="")
    print(", test loss: ", end="")
    print(['%.4f' % n for n in test_losses_list[-1]])

    if not os.path.isdir("save_dir"):
        os.mkdir("save_dir")
    torch.save(model.state_dict(), "save_dir/model_" + save_name + ".pth")
    np.save("save_dir/loss_train_" + save_name + ".npy", train_losses_list)
    np.save("save_dir/loss_test_" + save_name + ".npy", test_losses_list)






