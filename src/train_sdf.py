import os
import sys
import torch
import numpy as np


def find_best_gpu():
    # this function finds the GPU with most free memory.
    if 'linux' in sys.platform and torch.cuda.device_count() > 1:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        os.remove('tmp')
        gpu_id = np.argmax(memory_available).item()
        print("best gpu is %d with %0.1f Gb available space" %(gpu_id, memory_available[gpu_id]/1000))
        return gpu_id


def get_device(use_cpu):
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        gpu_id = find_best_gpu()
        if gpu_id:
            torch.cuda.set_device(gpu_id)
    return device


def forward_step(model, data, loss_funcs, device, training=True, **loss_kwargs):
    # bring data to GPU/CPU
    data = data.to(device)

    # prepare for training/evaluation
    if training:
        model.train()
    else:
        model.eval()

    # get output
    pred = model(data)
    losses = [func(pred, data, **loss_kwargs) for func in loss_funcs]
    return losses


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


def train_sdf(model, train_data, test_data, loss_funcs, n_epoch=500, print_every=25, use_cpu=False,
              save_name="", lr_0=0.001, step_size=50, gamma=0.5, **losses_params):
    device = get_device(use_cpu)
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    train_losses_list = []
    test_losses_list = []

    for epoch in range(n_epoch + 1):
        epoch_loss = []
        for data in train_data:
            optimizer.zero_grad()
            losses = forward_step(model, data, loss_funcs, device, **losses_params)
            epoch_loss.append([ll.item() for ll in losses])
            train_loss = sum(losses)
            train_loss.backward()
            optimizer.step()
        epoch_loss_mean = np.mean(epoch_loss, axis=0)
        train_losses_list.append(epoch_loss_mean)

        if epoch % print_every == 0:
            test_epoch_loss = []
            for data in test_data:
                losses = forward_step(model, data, loss_funcs, device, training=False)
                test_epoch_loss.append([ll.item() for ll in losses])
            test_epoch_loss_mean = np.mean(test_epoch_loss, axis=0)
            test_losses_list.append(test_epoch_loss_mean)
            print_and_save(model, epoch, train_losses_list, test_losses_list, optimizer, save_name)

        scheduler.step()


