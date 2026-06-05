import timm
from torchinfo import summary

net = timm.create_model('MSBDN-RDFF-mix', pretrained=True, num_classes=120)
print(summary(net, input_size=(128, 3, 224, 224)))