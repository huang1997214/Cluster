from DNN_model import Get_ResNet_50, Get_VGG_16
from utils import vertex_extract, open_image, process_list, load_data
import torch
import datetime
import ray
import gc
import psutil

def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

#Build the model
VGG_16 = Get_VGG_16()
VGG_16 = VGG_16
#Extract Layers
layer_list = vertex_extract(VGG_16)
print(layer_list)
'''
#Divide Virtual Machine
divid_index = 19
VM_1 = layer_list[0:divid_index]
VM_2 = layer_list[divid_index:len(layer_list)]
#
image_list = load_data()
starttime = datetime.datetime.now()
for img in image_list:
    img = torch.unsqueeze(img, 0)
    com_result = process_list(VM_1, img)
    res = process_list(VM_2, com_result)
    #res = VGG_16(img)
endtime = datetime.datetime.now()
running_time = endtime - starttime
print('Original Running Time', running_time)
ray.init()

@ray.remote
def process_VM_1(x):
    print('xxxxxxxxxxxx')
    return process_list(VM_1, x)

@ray.remote
def process_VM_2(x):
    print('aaaaaaaaaaaa')
    return process_list(VM_2, x)

starttime = datetime.datetime.now()
img = image_list[0]
img = torch.unsqueeze(img, 0)
com_result = ray.get(process_VM_1.remote(img))
auto_garbage_collect()
res = ray.get(process_VM_2.remote(com_result))
auto_garbage_collect()
ray.shutdown()
endtime = datetime.datetime.now()
running_time = endtime - starttime
print('2VM Running Time', running_time)
'''