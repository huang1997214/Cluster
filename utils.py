import cnn_finetune
import torchvision
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

def vertex_extract(model):
    '''
    :param model: torch model of a DNN
    :return: layer list
    '''
    layer_list = []
    for layer in model.named_modules():
        #Skip Same Special Layers in the cnn_finetune
        if isinstance(layer[1], cnn_finetune.contrib.torchvision.ResNetWrapper):
            continue
        if isinstance(layer[1], torch.nn.modules.container.Sequential):
            continue
        if isinstance(layer[1], torchvision.models.resnet.Bottleneck):
            continue
        if isinstance(layer[1], cnn_finetune.contrib.torchvision.VGGWrapper):
            continue
        else:
            layer_list.append(layer[1])
    return layer_list

def Generate_DAG_Graph(layer_list, model_name):
    assert model_name in ['ResNet18', 'ResNet50', 'VGG16', 'Inception_v3', 'Inception_v4']

def open_image(img_dir, img_resize=224):
    template_transform = transforms.Compose([
        transforms.Resize((img_resize, img_resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_dir).convert('RGB')
    img = template_transform(img)
    return img

drop_out_rate = 3
Relu_para = 4

class Layer():
    def __init__(self, layer_type, in_shape, out_shape, index, kernel_size, stride, dependence):
        self.layer_type = layer_type
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.index = index
        self.dependence = dependence
        assert self.layer_type in ['conv', 'pool', 'ReLu', 'Dropout', 'FC']
        if self.layer_type == 'ReLu':
            if len(in_shape)==4:
                self.memory = 2 * in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3] * 16
                self.cpu = Relu_para * in_shape[0] * in_shape[1] * in_shape[2]
                self.in_size = in_shape[1] * in_shape[2] * in_shape[3]
            else:
                self.memory = 2 * in_shape[0] * 16
                self.cpu = Relu_para * in_shape[0]
                self.in_size = in_shape[0]
        elif self.layer_type == 'FC':
            self.memory = (in_shape[0] + out_shape[0]) * 16
            self.cpu = 2 * in_shape[0] * out_shape[0]
            self.in_size = in_shape[0]
        elif self.layer_type == 'Dropout':
            self.memory = (in_shape[0] + out_shape[0]) * 16
            self.cpu = drop_out_rate * in_shape[0] * out_shape[0] + (in_shape[0] + out_shape[0])
            self.in_size = in_shape[0]
        elif self.layer_type == 'conv':
            self.memory = 2 * in_shape[1] * in_shape[2] * in_shape[3] * 16 \
                          + 2 * out_shape[1] * out_shape[2] * out_shape[3] * 16
            self.cpu = in_shape[1] * in_shape[2] * in_shape[3] * kernel_size**2 / (stride**2)
            self.in_size = in_shape[1] * in_shape[2] * in_shape[3]
        elif self.layer_type == 'pool':
            self.memory = 2 * in_shape[1] * in_shape[2] * in_shape[3] * 16 \
                          + 2 * out_shape[1] * out_shape[2] * out_shape[3] * 16
            self.cpu = in_shape[1] * in_shape[2] * in_shape[3] * kernel_size ** 2 / (stride ** 2)
            self.in_size = in_shape[1] * in_shape[2] * in_shape[3]

class VM():
    def __init__(self, cpu, memory, id):
        self.cpu = cpu
        self.memory = memory
        self.id = id
        self.assign_layer_id = []
        self.memory_weight = 1
        self.cpu_weight = 1
        self.cpu_cal = 0

    def assign_layer(self, layer):
        self.assign_layer_id.append(layer.index)
        self.cpu_cal += layer.cpu
        self.memory -= layer.memory

    def check_dependence(self, layer):
        if layer.dependence in self.assign_layer_id:
            return True
        else:
            return False

    def cal_distance(self, layer):
        if self.memory - layer.memory<0:
            return -1
        distance = self.memory_weight * (self.memory - layer.memory) \
                   + self.cpu_cal
        return distance

def evaluate_performance(vm_list, layer_list):
    bandwidth = 10000
    communication = 0
    running_time = 0
    for layer in layer_list:
        index = layer.index
        for vm in vm_list:
            if index in vm.assign_layer_id:
                #If need communication
                if index != 0:
                    if layer.dependence not in vm.assign_layer_id:
                        communication += layer.in_size
                        running_time += layer.in_size / bandwidth
                #Running
                running_time += layer.cpu / (vm.cpu * 1000)
    return running_time, communication