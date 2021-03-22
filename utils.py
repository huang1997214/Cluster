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

#TOBD
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