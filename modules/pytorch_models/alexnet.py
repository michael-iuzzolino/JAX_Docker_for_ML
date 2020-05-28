import torch
import torch.nn as nn

get_shape = lambda x : x.detach().numpy().shape

class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        self.activation = lambda x : nn.ReLU(inplace=True)(x)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):
        print("x:     ", get_shape(x))
        
        out = self.activation(self.conv1(x))
        print("conv1: ", get_shape(out))
        out = self.mp(out)
        print("mp1:   ", get_shape(out))
        
        out = self.activation(self.conv2(out))
        print("conv2: ", get_shape(out))
        out = self.mp(out)
        print("mp2:   ", get_shape(out))
        
        out = self.activation(self.conv3(out))
        print("conv3: ", get_shape(out))
        out = self.activation(self.conv4(out))
        print("conv4: ", get_shape(out))
        out = self.activation(self.conv5(out))
        print("conv5: ", get_shape(out))
        out = self.mp(out)
        print("mp3:   ", get_shape(out))
        
        return out

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = Features()
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        print("Feature Stack")
        x = self.features(x)
        x = self.avgpool(x)
        print("PostAP:", get_shape(x))
        x = torch.flatten(x, 1)
        print("Flat:  ", get_shape(x))
        print("Classifier Stack")
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    model = AlexNet(**kwargs)
    return model