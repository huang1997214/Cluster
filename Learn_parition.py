import cv2
import cnn_finetune
import torch
import numpy as np
from utils import vertex_extract, open_image
from VM_Model import VM
from Layer_Model import Layer

def generate_layer_list(res):
    img = open_image('1.jpg')
    img = torch.unsqueeze(img, 0)
    layer_list = []
    index = 0
    for layer in res:
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            temp_class = Layer('conv', img.shape, layer(img).shape, index, layer.kernel_size[0],
                               layer.stride[0], index - 1)
            img = layer(img)
        elif isinstance(layer, torch.nn.modules.activation.ReLU):
            temp_class = Layer('ReLu', img.shape, layer(img).shape, index, 0,
                               0, index - 1)
            img = layer(img)

        elif isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
            temp_class = Layer('pool', img.shape, layer(img).shape, index, layer.kernel_size,
                               layer.stride, index - 1)
            img = layer(img)
        elif isinstance(layer, torch.nn.modules.dropout.Dropout):
            temp_class = Layer('Dropout', img.shape, layer(img).shape, index, 0,
                               0, index - 1)
            img = layer(img)
        elif isinstance(layer, torch.nn.modules.linear.Linear):
            img = img.view(-1)
            temp_class = Layer('FC', img.shape, layer(img).shape, index, 0,
                               0, index - 1)

            img = layer(img)
        layer_list.append(temp_class)
        index += 1
    return layer_list