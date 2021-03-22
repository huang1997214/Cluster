from DNN_model import Get_ResNet_50, Get_VGG_16
from utils import vertex_extract
from VM_Model import VM
from Learn_parition import generate_layer_list
import cv2
import cnn_finetune
import torch
import numpy as np
from Approaches import ALD, RoundRobin, Random


#The model we used, in this experiment is VGG16
VGG_16 = Get_VGG_16()
#Extract each layer as a vertex
res = vertex_extract(VGG_16)
#Calculate memory/cpu requriements for each vertex
layer_list = generate_layer_list(res)
print(layer_list)
#Define VM
vm1 = VM(1e5, 8e8, 1)
vm2 = VM(1e5, 6e8, 2)
vm3 = VM(1e5, 6e8, 3)
vm_list = [vm1, vm2, vm3]

#Make the experiment
ALD(vm_list, layer_list)
RoundRobin(vm_list, layer_list)
Random(vm_list, layer_list)

