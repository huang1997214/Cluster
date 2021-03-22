import numpy as np
from utils import evaluate_performance


def ALD(vm_list, layer_list):
    for layer in layer_list:
        assigned = False
        if layer.dependence != -1:
            for vm in vm_list:
                if vm.check_dependence(layer) and vm.check_memory_penalty(layer):
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
    # Our policy Result
    print('###############Ours################')
    print('Assign Results:')
    for i in range(len(vm_list)):
        print(vm_list[i].assign_layer_id)
    # Evaluation
    running_time, communication = evaluate_performance(vm_list, layer_list)
    print('Total Running Time:')
    print(running_time)
    print('Communication Bytes:')
    print(communication)


def Random(vm_list, layer_list):
    for layer in layer_list:
        assigned = False
        while not assigned:
            vm_id = np.random.randint(0, 3)
            if vm_list[vm_id].cal_distance(layer) != -1:
                vm_list[vm_id].assign_layer(layer)
                assigned = True
    print('###############Random################')
    print('Assign Results:')
    for i in range(len(vm_list)):
        print(vm_list[i].assign_layer_id)
    # Evaluation
    running_time, communication = evaluate_performance(vm_list, layer_list)
    print('Total Running Time:')
    print(running_time)
    print('Communication Bytes:')
    print(communication)

def RoundRobin(vm_list, layer_list):
    index = 0
    for layer in layer_list:
        assigned = False
        while not assigned:
            vm_id = index % 3
            if vm_list[vm_id].cal_distance(layer) != -1:
                vm_list[vm_id].assign_layer(layer)
                assigned = True
        index += 1
    print('###############Round Robin################')
    print('Assign Results:')
    for i in range(len(vm_list)):
        print(vm_list[i].assign_layer_id)
    # Evaluation
    running_time, communication = evaluate_performance(vm_list, layer_list)
    print('Total Running Time:')
    print(running_time)
    print('Communication Bytes:')
    print(communication)