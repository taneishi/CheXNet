import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timeit
import sys
import os
from read_data import ChestXrayDataSet

MODEL_PATH = 'model/model.pth'
N_CLASSES = 14
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 32

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

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
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

    now = timeit.default_timer()

    # switch to evaluate mode
    model.eval()

    for inp, target in test_loader:
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).to(device))
            output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

    print('time: %0.2f sec.' % (timeit.default_timer() - now))

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.3f}'.format(CLASS_NAMES[i], AUROCs[i]))

def compute_AUCs(gt, pred):
    '''Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    '''
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

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
