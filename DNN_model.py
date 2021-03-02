from cnn_finetune import make_model


def Get_ResNet_50(classes = 10):
    model = make_model('resnet50', num_classes=classes, pretrained=True)
    return model

def Get_VGG_16(classes = 10):
    model = make_model('vgg16', num_classes=classes, pretrained=True, input_size=(224, 224))
    return model

def Get_Inception_v4(classes = 10):
    model = make_model('inception_v4', num_classes=classes, pretrained=True)
    return model

def Get_Inception_v3(classes = 10):
    model = make_model('inception_v3', num_classes=classes, pretrained=True)
    return model
