import torch.nn as nn
import torchvision

N_CLASSES = 14
CLASS_NAMES = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia']

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_features, out_size),
                nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x
