import torch
from model import DenseNet121, CLASS_NAMES, N_CLASSES
import argparse
import os

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize and load the model
    model = DenseNet121(N_CLASSES).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)

    if os.path.isfile(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('model state has loaded')
    else:
        print('=> model state file not found')

    model.train(False)
    dummy_input = torch.randn(args.batch_size, 3, 224, 224)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model/model.pth', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    main(args)
