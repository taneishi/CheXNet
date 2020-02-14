'''
The main CheXNet model implementation.
'''
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import logging as log
from openvino.inference_engine import IENetwork, IECore
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
import timeit
from time import time
import sys
import os

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
#TEST_IMAGE_LIST = './ChestX-ray14/labels/test_32.txt'
#TEST_IMAGE_LIST = './ChestX-ray14/labels/test_128.txt'
#TEST_IMAGE_LIST = './ChestX-ray14/labels/test_200.txt'
#TEST_IMAGE_LIST = './ChestX-ray14/labels/test_35.txt'
BATCH_SIZE = 32
N_CROPS = 10

def crop(img, top, left, height, width):
    return img.crop((left,top, left+width, top+height))

def five_crop(img, size):
    image_width, image_height = img.width
    crop_heigh, crop_width = size
    tl = img.crop((0,0, crop_width, crop_height))
    tr = img.crop((image_width - crop_width, 0, image_width, crop_height))
    return (tl, tr)
    
def main():
    model_xml = 'densenet121_i8.xml'
    model_bin = os.path.splitext(model_xml)[0]+'.bin'

    log.info('Creating Inference Engine')
    ie = IECore()
    net = IENetwork(model=model_xml, weights=model_bin)
    log.info('Preparing input blobs')
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = (BATCH_SIZE*N_CROPS)

    n, c, h, w = net.inputs[input_blob].shape

    # for image load
    # test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
    #       image_list_file=TEST_IMAGE_LIST,
    #       transform=transforms.Compose([
    #           transforms.Resize(256),
    #           transforms.TenCrop(224),
    #           transforms.Lambda
    #           (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))]))
    normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
            image_list_file=TEST_IMAGE_LIST,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda
                (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda
                (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                ]))
    
    print(test_dataset)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=10, pin_memory=False)

    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    
    # images = np.ndarray(shape=(n,c,h,w))

    #loading model to the plugin
    log.info('Loading model to the plugin')
    #exec_net = ie.load_network(network=net, device_name='CPU', config={'DYN_BATCH_ENABLED': 'YES'})
    exec_net = ie.load_network(network=net, device_name='CPU')

    now = timeit.default_timer()
    for i , (inp, target) in enumerate(test_loader):
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        images = inp.view(-1, c, h, w).numpy()
        # print(images.shape)
        # print(bs)
        if bs !=  BATCH_SIZE:
            images2 = np.zeros(shape=(BATCH_SIZE* n_crops, c, h, w))
            images2[:bs*n_crops, :c, :h, :w] = images
            images = images2
        res = exec_net.infer(inputs={input_blob: images})
        res = res[out_blob]
        res = res.reshape(BATCH_SIZE, n_crops,-1)
        #print(res)
        res = np.mean(res, axis=1)
        if bs != BATCH_SIZE:
            #print(res.shape)
            res = res[:bs, :res.shape[1]]
        #print(res)
        pred = torch.cat((pred, torch.from_numpy(res)), 0)
        #print(res.shape)
        
    print('Elapsed time: %0.2f sec.' % (timeit.default_timer() - now))

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.3f}'.format(CLASS_NAMES[i], AUROCs[i]))

def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)
        
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
        AUROCs.append(roc_auc_score_FIXED(gt_np[:, i], pred_np[:, i]))
    return AUROCs

if __name__ == '__main__':
    main()
