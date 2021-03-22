class VM():
    def __init__(self, cpu, memory, id):
        self.cpu = cpu
        self.memory = memory
        self.id = id
        self.assign_layer_id = []
        self.memory_weight = 1
        self.memory_penalty = 100
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

    def check_memory_penalty(self, layer):
        if self.memory - layer.memory < 0:
            return False
        else:
            return True

    def calculate_dependence(self, layer):
        dep = 0
        if layer.dependence in self.assign_layer_id:
            dep += 1
        return dep

    def cal_distance(self, layer):
        if self.memory - layer.memory<0:
            distance = self.memory_penalty + self.cpu_cal
        else:
            distance = self.memory_weight * (self.memory - layer.memory) \
                       + self.cpu_cal
        return distance