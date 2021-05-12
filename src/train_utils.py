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


def get_device(device):
    assert isinstance(device, list)

    if len(device) == 1 and device[0] == 'cpu':
        print('training using cpu')
        device = torch.device('cpu')
    elif len(device) == 1 and device[0] == 'cuda':
        print('training using gpu: ', end="")
        device = torch.device('cuda')
        gpu_id = find_best_gpu()
        if gpu_id:
            torch.cuda.set_device(gpu_id)
    elif isinstance(device, list) and all(isinstance(x, int) for x in device):
        device = [int(x) for x in device]
        print('training using multiple gpus: ', device)
    else:
        raise ValueError("device is not correct.")
    return device


def train_forward_step(model, data, loss_funcs, device, training=True, **loss_kwargs):
    if not isinstance(device, list):
        data = data.to(device)
    if training:
        model.train()
    else:
        model.eval()

    pred = model(data)
    losses = [func(pred, data, **loss_kwargs) for func in loss_funcs]
    return losses
