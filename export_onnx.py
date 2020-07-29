import torch
from model import DenseNet121
import os

MODEL_PATH = 'model/model.pth'
N_CLASSES = 14
CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
BATCH_SIZE = 32

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize and load the model
    model = DenseNet121(N_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)

    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print('model state has loaded')
    else:
        print('=> model state file not found')

    model.train(False)
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
    torch_out = model(dummy_input)
    torch.onnx.export(model,
            dummy_input, 'model/densenet121.onnx',
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0 : 'batch_size'},
                'output': {0: 'batch_size'}},
            verbose=False)
    print('ONNX model exported.')

if __name__ == '__main__':
    main()
