import numpy as np
import torchvision
from torchvision import transforms
import torch
from sklearn.metrics import roc_auc_score
import argparse
import timeit
import os

from datasets import ChestXrayDataSet
from model import DenseNet121, CLASS_NAMES, N_CLASSES

def init_distributed_mode(args):
    if args.hpu:
        import habana_frameworks.torch.core.hccl

    world_size = int(os.environ[args.env_world_size])
    local_rank = int(os.environ[args.env_rank])

    print('distributed init (rank {})'.format(local_rank), flush=True)

    if args.hpu:
        os.environ['ID'] = str(local_rank)
        # Not used currently.
        os.environ['LOCAL_RANK'] = str(local_rank)
        backend = 'hccl'
    else:
        torch.cuda.set_device(local_rank)
        backend = 'nccl'

    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=local_rank)

def main(args):
    torch.manual_seed(123)
    torch.multiprocessing.set_start_method('spawn')

    world_size = int(os.environ[args.env_world_size])
    local_rank = int(os.environ[args.env_rank])

    if local_rank == 0:
        print(vars(args))

    if args.hpu:
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        import habana_frameworks.torch.core as htcore
        load_habana_module()
        device = torch.device('hpu')

        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

        os.environ['PT_HPU_LAZY_MODE'] = '1'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if world_size > 1:
        init_distributed_mode(args)

    print('Using %s device.' % (device))

    normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])

    transform = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])

    train_dataset = ChestXrayDataSet(
            data_dir=args.data_dir,
            image_list_file=args.train_image_list,
            transform=transform,
            )

    train_sampler = None
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # Argument sampler is mutually exclusive with shuffle.
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler)

    if local_rank == 0:
        print('training %d batches %d images' % (len(train_loader), len(train_dataset)))

    # Initialize and load the model.
    net = DenseNet121(N_CLASSES)

    if args.model_path:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        if local_rank == 0:
            print('model state has loaded.')

    net = net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    if world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, 
                bucket_cap_mb=100, 
                broadcast_buffers=False, 
                gradient_as_bucket_view=True)

    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.MultiLabelSoftMarginLoss()

    for epoch in range(args.epochs):
        start_time = timeit.default_timer()

        if world_size > 1:
            train_sampler.set_epoch(epoch)

        # Initialize the ground truth and output tensor.
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        train_loss = 0
        net.train()
        for index, (images, labels) in enumerate(train_loader, 1):
            # Each image has 10 crops.
            batch_size, n_crops, c, h, w = images.size()
            images = images.view(-1, c, h, w).to(device)
            labels = labels.to(device)

            outputs = net(images)

            outputs_mean = outputs.view(batch_size, n_crops, -1).mean(1)
            loss = criterion(outputs_mean, labels)
            train_loss += loss.item()

            y_true = torch.cat((y_true, labels.detach().cpu()))
            y_pred = torch.cat((y_pred, outputs_mean.detach().cpu()))

            # Backward and optimize.
            optimizer.zero_grad()
            loss.backward()
            if args.hpu:
                htcore.mark_step()
            optimizer.step()
            if args.hpu:
                htcore.mark_step()

            if local_rank == 0:
                print('\repoch {:3d}/{:3d} batch {:5d}/{:5d}'.format(epoch+1, args.epochs, index, len(train_loader)), end='')
                print(' train loss {:6.4f}'.format(train_loss / index), end='')
                print(' {:6.3f}sec'.format(timeit.default_timer() - start_time), end='')

                aucs = [roc_auc_score(y_true[:, i], y_pred[:, i]) if y_true[:, i].sum() > 0 else np.nan for i in range(N_CLASSES)]
                auc_classes = ' '.join(['{:5.3f}'.format(aucs[i]) for i in range(N_CLASSES)])
                print(' average AUC {:5.3f}'.format(np.mean(aucs)), end='')

        if local_rank == 0:
            print('')
            torch.save(net.state_dict(), 'model/checkpoint.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_world_size', default='WORLD_SIZE', type=str)
    parser.add_argument('--env_rank', default='LOCAL_RANK', type=str)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--hpu', action='store_true', default=False)
    parser.add_argument('--use_lazy_mode', action='store_true', default=True)
    parser.add_argument('--train_image_list', default='labels/train_list.txt', type=str)
    args = parser.parse_args()

    main(args)
