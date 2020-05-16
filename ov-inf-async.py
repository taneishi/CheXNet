'''
The main CheXNet model implementation.
'''
import numpy as np
from sklearn.metrics import roc_auc_score
from openvino.inference_engine import IENetwork, IECore
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
import logging as log
import timeit
import os

# for async
#from openvino.tools.benchmark.utils.infer_request_wrap import InferRequestsQueue
# copy code from the file
from datetime import datetime
import threading

class InferReqWrap:
    def __init__(self, request, req_id, callback_queue, out_blob):
        self.req_id = req_id
        self.request = request
        self.request.set_completion_callback(self.callback, self.req_id)
        self.callbackQueue = callback_queue
        self.__ground_truth = torch.FloatTensor()
        self.__pred = torch.FloatTensor()
        self.out_blob = out_blob

    def callback(self, status_code, user_data):
        if user_data != self.req_id:
            print('Request ID {} does not correspond to user data {}'.format(self.req_id, user_data))
        elif status_code:
            print('Request {} failed with status code {}'.format(self.req_id, status_code))
        res = self.request.outputs[self.out_blob]
        res = res.reshape(BATCH_SIZE, N_CROPS, -1)
        res = np.mean(res, axis = 1)
        if self.__bs != BATCH_SIZE:
            res = res[:self.__bs, :res.shape[1]]
        self.__pred = torch.cat((self.__pred, torch.from_numpy(res)), 0)
        self.callbackQueue(self.req_id, self.request.latency)

    def start_async(self, input_data, bs, ground_truth=None):
        self.__ground_truth=torch.cat((self.__ground_truth, ground_truth), 0)
        self.__bs = bs
        self.request.async_infer(input_data)

    def infer(self, input_data, ground_truth=None):
        self.request.infer(input_data)
        self.callbackQueue(self.req_id, self.request.latency)

    def get_ground_truth(self):
        return self.__ground_truth

    def get_prediction(self):
        return self.__pred

class InferRequestsQueue:
    def __init__(self, requests, out_blob):
        self.idleIds = []
        self.requests = []
        self.times = []
        for req_id in range(len(requests)):
            self.requests.append(InferReqWrap(requests[req_id], req_id, self.put_idle_request, out_blob))
            self.idleIds.append(req_id)
        self.startTime = datetime.max
        self.endTime = datetime.min
        self.cv = threading.Condition()

    def reset_times(self):
        self.times.clear()

    def get_duration_in_seconds(self):
        return (self.endTime - self.startTime).total_seconds()

    def put_idle_request(self, req_id, latency):
        self.cv.acquire()
        self.times.append(latency)
        self.idleIds.append(req_id)
        self.endTime = max(self.endTime, datetime.now())
        self.cv.notify()
        self.cv.release()

    def get_idle_request(self):
        self.cv.acquire()
        while len(self.idleIds) == 0:
            self.cv.wait()
        req_id = self.idleIds.pop()
        self.startTime = min(datetime.now(), self.startTime)
        self.cv.release()
        return self.requests[req_id]

    def wait_all(self):
        self.cv.acquire()
        while len(self.idleIds) != len(self.requests):
            self.cv.wait()
        self.cv.release()

N_CLASSES = 14
CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
DATA_DIR = './ChestX-ray14/images'
TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
BATCH_SIZE = 32
N_CROPS = 10
NUM_REQUESTS=8

def main():
    #model_xml = 'model/densenet121_i8.xml'
    model_xml = 'model/densenet121.xml'
    model_bin = os.path.splitext(model_xml)[0]+'.bin'

    log.info('Creating Inference Engine')
    ie = IECore()
    net = IENetwork(model=model_xml, weights=model_bin)
    log.info('Preparing input blobs')
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = (BATCH_SIZE*N_CROPS)

    n, c, h, w = net.inputs[input_blob].shape

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
    
    #loading model to the plugin
    log.info('Loading model to the plugin')
    #exec_net = ie.load_network(network=net, device_name='CPU', config={'DYN_BATCH_ENABLED': 'YES'})

    #config = {'CPU_THREADS_NUM': '48', 'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
    config = {'CPU_THROUGHPUT_STREAMS': '8'}
    exec_net = ie.load_network(network=net, device_name='CPU', config=config, num_requests=NUM_REQUESTS)
    # Number of requests
    infer_requests = exec_net.requests
    print('reqeuest len', len(infer_requests))
    request_queue = InferRequestsQueue(infer_requests, out_blob)

    for i, (inp, target) in enumerate(test_loader):
        start_time = timeit.default_timer()
        # gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        images = inp.view(-1, c, h, w).numpy()
        #print(images.shape)
        #print(bs)
        if bs !=  BATCH_SIZE:
            images2 = np.zeros(shape=(BATCH_SIZE * n_crops, c, h, w))
            images2[:bs*n_crops, :c, :h, :w] = images
            images = images2

        infer_request = request_queue.get_idle_request()
        # print(infer_request.request)
        # Infer async

        infer_request.start_async({input_blob: images}, bs, target)
        
        # res = res[out_blob]
        # res = res.reshape(BATCH_SIZE, n_crops,-1)
        # #print(res)
        # res = np.mean(res, axis=1)
        # if bs != BATCH_SIZE:
        #     print(res.shape)
        #     res = res[:bs, :res.shape[1]]
        # #print(res)
        # pred = torch.cat((pred, torch.from_numpy(res)), 0)
        # #print(res.shape)

    # wait the latest inference executions
    request_queue.wait_all()
    for i, queue in enumerate(request_queue.requests):
        # print(i, queue)
        gt = torch.cat((gt, queue.get_ground_truth()), 0)
        pred = torch.cat((pred, queue.get_prediction()), 0)
        
    print('Elapsed time: %0.2f sec.' % (timeit.default_timer() - now))

    AUCs = [roc_auc_score(gt.cpu()[:, i], pred.cpu()[:, i]) for i in range(N_CLASSES)]
    AUC_avg = np.array(AUCs).mean()

    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {:.3f}'.format(CLASS_NAMES[i], AUROCs[i]))

if __name__ == '__main__':
    main()
