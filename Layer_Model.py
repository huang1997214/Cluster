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