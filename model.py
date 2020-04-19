import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
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
BATCH_SIZE = 4
MODEL_EXPORT = True

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)

    # switch to evaluate mode
    model.eval()

    for index, (data, target) in enumerate(test_loader):
        start_time = timeit.default_timer()
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = data.size()
        with torch.no_grad():
            input_var = torch.autograd.Variable(data.view(-1, c, h, w).to(device))
            output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

        print('%03d/%03d, time: %6.3f sec' % (index, len(test_loader), (timeit.default_timer() - start_time)))

        if MODEL_EXPORT:
            torch.onnx.export(model,
                    input_var, 'model/densenet121.onnx',
                    export_params=True,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                    verbose=False)
            print('ONNX model exported.')
            sys.exit()
            
    AUCs = compute_AUCs(gt, pred)
    AUC_avg = np.array(AUCs).mean()
    print('The average AUC is {AUC_avg:.3f}'.format(AUC_avg=AUC_avg))
    for i in range(N_CLASSES):
        print('The AUC of {} is {:.3f}'.format(CLASS_NAMES[i], AUCs[i]))

def compute_AUCs(gt, pred):
    '''Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of ROC-AUCs of all classes.
    '''
    AUCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUCs

class DenseNet121(nn.Module):
    '''Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    '''
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.densenet121(x)
        return x

if __name__ == '__main__':
    main()
