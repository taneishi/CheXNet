# The main CheXNet model implementation.
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import logging as log
from openvino.inference_engine import IENetwork, IECore
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
import timeit
import sys
import os

N_CLASSES = 14
CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'

N_CROPS = 10

def main(modelfile):
    batch_size = 32

    model_xml = os.path.join('model', modelfile)
    model_bin = model_xml.replace('xml', 'bin')

    log.info('Creating Inference Engine')
    ie = IECore()
    #net = IENetwork(model=model_xml, weights=model_bin)
    net = ie.read_network(model=model_xml, weights=model_bin)

    log.info('Preparing input blobs')
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = (batch_size * N_CROPS)

    n, c, h, w = net.inputs[input_blob].shape

    # for image load
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
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
                ]))
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    
    # images = np.ndarray(shape=(n,c,h,w))

    # loading model to the plugin
    log.info('Loading model to the plugin')
    #exec_net = ie.load_network(network=net, device_name='CPU', config={'DYN_BATCH_ENABLED': 'YES'})
    exec_net = ie.load_network(network=net, device_name='CPU')

    for index, (data, target) in enumerate(test_loader):
        start_time = timeit.default_timer()

        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = data.size()

        images = data.view(-1, c, h, w).numpy()

        if bs != batch_size:
            images2 = np.zeros(shape=(batch_size * n_crops, c, h, w))
            images2[:bs*n_crops, :c, :h, :w] = images
            images = images2

        res = exec_net.infer(inputs={input_blob: images})
        res = res[out_blob]
        res = res.reshape(batch_size, n_crops,-1)
        res = np.mean(res, axis=1)

        if bs != batch_size:
            res = res[:bs, :res.shape[1]]

        pred = torch.cat((pred, torch.from_numpy(res)), 0)
        
        print('%03d/%03d, time: %6.3f sec' % (index, len(test_loader), (timeit.default_timer() - start_time)))
        
    AUCs = [roc_auc_score_FIXED(gt.cpu()[:, i], pred.cpu()[:, i]) for i in range(N_CLASSES)]
    AUC_avg = np.array(AUCs).mean()
    print('The average AUC is {AUC_avg:.3f}'.format(AUC_avg=AUC_avg))
    for i in range(N_CLASSES):
        print('The AUC of {} is {:.3f}'.format(CLASS_NAMES[i], AUCs[i]))

def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)
        
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'fp32':
            main(modelfile='densenet121.xml')
        elif sys.argv[1] == 'int8':
            main(modelfile='densenet121_i8.xml')
        else:
            sys.exit('%s [fp32|int8]' % sys.argv[0])
    else:
        sys.exit('%s [fp32|int8]' % sys.argv[0])
