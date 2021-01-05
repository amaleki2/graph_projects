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
