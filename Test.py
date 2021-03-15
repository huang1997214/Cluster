from DNN_model import Get_ResNet_50, Get_VGG_16
from utils import vertex_extract, open_image, Layer, VM, evaluate_performance
import cv2
import cnn_finetune
import torch
import numpy as np
img = open_image('1.jpg')
img = torch.unsqueeze(img, 0)
VGG_16 = Get_VGG_16()
res = vertex_extract(VGG_16)
#Divide Index
divid_index = 19

index = 0
layer_list = []
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

vm1 = VM(1e5, 8e8, 1)
vm2 = VM(1e5, 6e8, 2)
vm3 = VM(1e5, 6e8, 3)
vm_list = [vm1, vm2, vm3]

for layer in layer_list:
    assigned = False
    if layer.dependence != -1:
        for vm in vm_list:
            if vm.check_dependence(layer):
                if vm.cal_distance(layer) != -1:
                    vm.assign_layer(layer)
                    assigned = True
    if not assigned:
        dis_list = []
        for vm in vm_list:
            dis_list.append(vm.cal_distance(layer))
        dis = max(dis_list)
        dis_max = np.argmax(np.array(dis_list))
        if dis == -1:
            print('Failed!!!!!')
        vm_list[dis_max].assign_layer(layer)
#Our policy Result
for i in range(len(vm_list)):
    print(vm_list[i].assign_layer_id)
#Evaluation
running_time, communication = evaluate_performance(vm_list, layer_list)
print(running_time)
print(communication)

#Random policy Result
#init VM
vm1 = VM(1e5, 8e8, 1)
vm2 = VM(1e5, 6e8, 2)
vm3 = VM(1e5, 6e8, 3)
vm_list = [vm1, vm2, vm3]
for layer in layer_list:
    assigned = False
    while not assigned:
        vm_id = np.random.randint(0,3)
        if vm_list[vm_id].cal_distance(layer) != -1:
            vm_list[vm_id].assign_layer(layer)
            assigned = True
for i in range(len(vm_list)):
    print(vm_list[i].assign_layer_id)
#Evaluation
running_time, communication = evaluate_performance(vm_list, layer_list)
print(running_time)
print(communication)