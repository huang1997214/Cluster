import cnn_finetune
import torchvision
import torch
from PIL import Image
from torchvision import transforms
import os

def vertex_extract(model):
    '''
    :param model: torch model of a DNN
    :return: layer list
    '''
    layer_list = []
    index = 0
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
            layer_list.append([index, layer[1]])
            index += 1
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

def load_data():
    path = os.path.join(os.getcwd(), 'query_image')
    image_dir_list = os.listdir(path)
    for i in range(len(image_dir_list)):
        image_dir_list[i] = os.path.join(path, image_dir_list[i])
    image_dir_list = image_dir_list[0:1]
    image_list = []
    for dir in image_dir_list:
        image_list.append(open_image(dir))
    return image_list

def process_list(list_layer, x):
    for _layer in list_layer:
        index, layer = _layer
        x = layer(x)
        if index == 30:
            x = x.view(x.size(0), -1)
    return x