import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
import argparse
import timeit
import sys
import os

MODEL_PATH = 'model/model.pth'
N_CLASSES = 14
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'

def export_onnx(model):
    torch.onnx.export(model,
            input_var, os.path.join('model', 'densenet121.onnx'),
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'}, 
                'output': {0: 'batch_size'}},
            verbose=False)
    sys.exit('ONNX model exported.')

def main(args):
    if args.export_model:
        device = torch.device('cpu')
        batch_size = 1
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 48
    print('Using %s device.' % device)
    
    # initialize and load the model
    model = DenseNet121(N_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)

    if os.path.isfile(MODEL_PATH):
        print('=> loading model state')
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print('=> loaded model state')
    else:
        print('=> no model state file found')

    normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
            image_list_file=TEST_IMAGE_LIST,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)

    # switch to evaluate mode
    model.eval()

    for index, (data, target) in enumerate(test_loader):
        start_time = timeit.default_timer()

        target = target.to(device)
        bs, n_crops, c, h, w = data.size()
        data = data.view(-1, c, h, w).to(device)

        with torch.no_grad():
            output = model(data)

        output_mean = output.view(bs, n_crops, -1).mean(1)

        gt = torch.cat((gt, target))
        pred = torch.cat((pred, output_mean))

        print('\rbatch %03d/%03d %6.3f sec' % (index, len(test_loader), (timeit.default_timer() - start_time)))

        if args.export_model:
            export_onnx(model)

        if index == 32:
            break
            
    AUCs = [roc_auc_score(gt.cpu()[:, i], pred.cpu()[:, i]) for i in range(N_CLASSES)]
    AUC_avg = np.mean(AUCs)
    print('The average AUC is %6.3f' % AUC_avg)

    for i in range(N_CLASSES):
        print('The AUC of %s is %6.3f' % (CLASS_NAMES[i], AUCs[i]))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-export_model', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
