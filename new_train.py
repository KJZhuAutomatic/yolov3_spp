import os
import pathlib
import datetime
import argparse
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from reimplement import YoloV3SPP, YoloDataset
from reimplement.train_utils.train_eval_utils import train_one_epoch, evaluate
from reimplement.train_utils.coco_utils import get_coco_api_from_dataset

def main(args):
    log_folder = args.log
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    log_folder = pathlib.Path(log_folder)
    log_file = str(log_folder / datetime.datetime.now().__format__('%Y-%m-%d-%H:%M')) + '.txt'
    
    grid_size = 32
    img_size = args.img_size
    if args.multi_scale:
        max_size = int(np.ceil(img_size / 0.667 / grid_size) * grid_size)
        min_size = int(np.ceil(img_size / 1.5 / grid_size) * grid_size)
        print('Use multi scale train, image size range [%d %d].'%(min_size, max_size))
        
    train_hyp = args.train_hyp
    with open(train_hyp) as f:
        train_hyp = yaml.load(f, Loader=yaml.FullLoader)
    train_hyp['obj'] *= (img_size / 320)

    data = args.data
    data_dict = {}
    with open(data) as f:
        for line in f.read().splitlines():
            key, val = line.split('=')
            if val.isnumeric():
                val = int(val)
            data_dict[key] = val
    
    num_cls = data_dict['classes']
    train_hyp['cls'] *= (num_cls / 80)

    model_cfg = args.model_cfg
    model = YoloV3SPP(model_cfg)
    device = torch.device(args.device)
    model.to(device)
    model.hyp = train_hyp
    param_group = []
    if args.freeze_layer:
        for idx, module in enumerate(model.module_list):
            if idx+1 not in model.yolo_layers: # next layer is not yolo layer
                for p in module.parameters():
                    p.requires_grad_(False)
            else:
                param_group = param_group + list(module.parameters())
            '''
            if module.__class__.__name__ == 'MaxPool2d':
                print(idx)
            78 80 82
            freeze layer above max pool when freeze_layer is False
            '''
    else:
        max_pool_ind = 78
        for module in model.module_list[:max_pool_ind]:
            for p in module.parameters():
                p.requires_grad_(False)
        param_group = list(model.module_list[max_pool_ind:].parameters())
    
    lr = train_hyp['lr0']
    momentum = train_hyp['momentum']
    weight_decay = train_hyp['weight_decay']
    optimizer = torch.optim.SGD(param_group, lr=lr, momentum=momentum, 
                                weight_decay=weight_decay, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs,
                                           eta_min=train_hyp['lr0']*train_hyp['lrf'])
    '''
    import matplotlib.pyplot as plt
    lr = []
    for _ in range(args.epochs):
        lr.append(lr_scheduler.get_last_lr())
        lr_scheduler.step()
    print(lr[-1])
    plt.plot(lr)
    plt.show()
    '''
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.weights.endswith('.pt'):
        ckpt = torch.load(args.weights, map_location=device)

        model_state_dict = model.state_dict()
        model_weight = {k: v for k, v in ckpt['model'].items() \
                        if model_state_dict.get(k).numel() == v.numel() }
        model.load_state_dict(model_weight, strict=False)

        start_epochs = ckpt['epochs'] + 1 if ckpt.get('epochs') is not None else 0

        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        
        if ckpt.get('scaler') is not None and scaler is not None:
            scaler.load_state_dict(ckpt['scaler'])

        if ckpt.get('log') is not None:
            with open(ckpt.get('log')) as origin_f:
                with open(log_file, 'w') as f:
                    f.write(origin_f.read())

        if ckpt.get('lr_scheduler') is not None:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    else:
        raise ValueError('weights should be ends with .pt get %s'%args.weights[-3:])

    trainset = YoloDataset(data_dict['train'], img_size=max_size, 
                           batch_size=args.batch_size, augment=True, 
                           rect=args.rect, aug_param=train_hyp)

    testset = YoloDataset(data_dict['valid'], img_size=args.img_size, 
                          batch_size=args.batch_size, augment=False, rect=True)

    num_worker = min(os.cpu_count(), args.batch_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=not args.rect, collate_fn=trainset.collate_fn, num_workers=num_worker)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                   collate_fn=testset.collate_fn, num_workers=num_worker)
    coco = get_coco_api_from_dataset(testloader.dataset)
    accumulate = max(1, 64 // args.batch_size)
    gs = 32
    writer = SummaryWriter(log_dir='runs')

    for epoch in range(start_epochs, args.epochs):

        mloss, lr = train_one_epoch(model, optimizer, trainloader, device, epoch, 
                        print_freq=100, accumulate=accumulate, img_size=img_size,
                        gs=gs, grid_min=min_size//gs, grid_max=max_size//gs,
                        multi_scale=args.multi_scale, warmup=True, scaler=scaler)

        lr_scheduler.step()

        result_info = evaluate(model, testloader, coco=coco, device=device)
        text_ls = [str(i) for i in result_info] + [str(lr)]
        with open(log_file, 'a') as f:
            text = f'epoch [{epoch}/{args.epochs}]: ' + '  '.join(text_ls)
            f.write(text)

        coco_mAP = result_info[0]
        voc_mAP = result_info[1]
        tb_vals = [coco_mAP, voc_mAP] + mloss.detach().cpu().numpy().tolist()
        tb_tags = ['coco_mAP', 'voc_mAP', 'train/box_loss', 'train/obj_loss', 'train/class_loss', 'train/loss']
        for tag, val in zip(tb_tags, tb_vals):
            writer.add_scalar(tag, val, epoch)

        # save all in checkpoint
        save_dict = {'model': model.state_dict(),
                     'epochs': epoch,
                     'optimizer': optimizer.state_dict(),
                     'log': log_file,
                     'lr_scheduler': lr_scheduler.state_dict()}
        if scaler is not None:
            save_dict['scaler'] = scaler.state_dict()
        weight_dir = pathlib.Path(args.weights).parent
        save_file = str(weight_dir / f'yolov3-spp-epoch{epoch}-{args.epochs}.pt')
        torch.save(save_dict, save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hyperparameters.')
    parser.add_argument('--log', default='./log', type=str, 
                        help='path to log, default: ./log')
    parser.add_argument('--img-size', default=512, type=int, 
                        help='size of image, default: 512')
    parser.add_argument('--multi-scale', default=True, type=bool, 
                        help='start multi scale train, default: True')
    parser.add_argument('--train-hyp', default='./cfg/train_hyp.yaml', type=str, 
                        help='hyperparameters in train, default: ./cfg/train_hyp.yaml')
    parser.add_argument('--data', default='./data/my_data.data', type=str,
                        help='data path dict.')
    parser.add_argument('--model-cfg', default='./cfg/my_yolov3.cfg', type=str,
                        help='model construction config file')
    parser.add_argument('--device', default='cuda:0', type=str, help='')
    parser.add_argument('--freeze-layer', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--weights', default='./weights/yolov3-spp-ultralytics-512.pt',
                        type=str, help='file of weights')
    parser.add_argument('--epochs', default=30, type=int,
                        help='num of epochs to train and evaluate.')
    parser.add_argument('--batch-size', default=16, type=int, help='')
    parser.add_argument('--rect', action='store_true')
    args = parser.parse_args()
    main(args)
