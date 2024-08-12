import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class AudioResNet152(nn.Module):
    def __init__(self, num_classes):
        super(AudioResNet152, self).__init__()
        # Load pre-trained ResNet-152 model
        self.resnet = models.resnet152(models.ResNet152_Weights.DEFAULT)
        # Modify first layer to accept 1-channel input (for grayscale spectrogram/MFCC)
        # Assumes input size of 224x224 (adjust if your input size is different)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify output layer to match number of classes
        # resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.resnet.fc = nn.Linear(2048, num_classes)
        # self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

class AudioEfficientNetB7(nn.Module):
  def __init__(self, num_classes):
    super(AudioEfficientNetB7, self).__init__()
    # Load pre-trained EfficientNet-B7 model
    self.efficientnet = models.efficientnet_b7(models.EfficientNet_B7_Weights.DEFAULT)
    # Modify first layer to accept 1-channel input (for grayscale spectrogram/MFCC)
    # Assumes input size of 224x224 (adjust if your input size is different)
    # self.efficientnet.conv1  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # self.efficientnet._conv_stem = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
    self.efficientnet.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # Access classifier module and modify final layer
    # self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, num_classes)
    self.efficientnet.classifier[1] = nn.Linear(in_features=2560, out_features=num_classes)
    # self.efficientnet = efficientnet

  def forward(self, x):
    return self.efficientnet(x)


class AlexNet(nn.Module):

    def __init__(self, num_classes=6):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AudioAlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AudioAlexNet, self).__init__()
        # Load pre-trained AlexNet model
        alexnet = models.alexnet(models.AlexNet_Weights.DEFAULT)
        # Modify first layer to accept 1-channel input (for grayscale spectrogram/MFCC)
        # Assumes input size of 224x224 (adjust if your input size is different)
        alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        # Modify output layer to match number of classes
        alexnet.classifier[6] = nn.Linear(4096, num_classes)
        self.alexnet = alexnet

    def forward(self, x):
        return self.alexnet(x)

class M5(nn.Module):
    def __init__(self, n_input=1, num_classes=2, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)



class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        
        self.model = models.densenet201(models.DenseNet201_Weights.DEFAULT)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier = nn.Linear(1920, num_classes)
        
    def forward(self, x):
        output = self.model(x)
        return output
    
class Inception(nn.Module):
    def __init__(self,num_classes ):
        super(Inception, self).__init__()
        
        self.model = models.inception_v3(models.Inception_V3_Weights.DEFAULT)
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output
     
class ResNet50(nn.Module):
	def __init__(self, num_classes):
		super(ResNet50, self).__init__()
		
		self.model = models.resnet50(models.ResNet50_Weights.DEFAULT)
		self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.model.fc = nn.Linear(2048, num_classes)
	def forward(self, x):
		output = self.model(x)
		return output

if __name__ == "__main__":
    print(Inception(num_classes=2))