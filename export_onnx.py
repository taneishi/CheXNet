'''
Export ONNX model
'''
import torch
import torch.nn as nn
import torchvision
import os

MODEL_PATH = 'model/model.pth'
N_CLASSES = 14
DATA_DIR = './ChestX-ray14/images'
BATCH_SIZE = 32

def main():
    device = torch.device('cpu')
    
    # initialize and load the model
    model = DenseNet121(N_CLASSES).to(device)

    if os.path.isfile(MODEL_PATH):
        print('=> loading model state')
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print('=> loaded model state')
    else:
        print('=> no model state file found')

    model.train(False)
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
    torch_out = model(dummy_input)
    torch.onnx.export(model,
            dummy_input, 'model/densenet121.onnx',
            export_params=True,
            do_constant_folding= True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=True)
    print('ONNX model exported.')

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
