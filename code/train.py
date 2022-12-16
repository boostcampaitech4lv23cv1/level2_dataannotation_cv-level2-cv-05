import os
import os.path as osp
import time
import math
import datetime
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import wandb

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    now = (datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=9)).strftime("%m-%d %H:%M")
    wandb.init(project="data_anno", entity="aidatapkt", name = f'{now}')
    wandb.config = {
        "epoch" : max_epoch,
        "lr" : learning_rate,
        "image_size": image_size
                    }
    
    dataset = SceneTextDataset(data_dir, split='final_icdar17_19_base_camper', image_size=image_size, crop_size=input_size, color_jitter=True)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # val추가
    val_dataset = SceneTextDataset(data_dir, split='camper_val', image_size=image_size, crop_size=input_size, color_jitter=False)
    val_dataset = EASTDataset(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.load_state_dict(torch.load('pretrained_model/latest.pth')) # pretrained model load
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 46, 66], gamma=0.1) ## 55부터 이어서 하기에
    
    min_loss = 1000
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        loss, val_3total_mean_loss, cls_loss, angle_loss, iou_loss = validation(model, val_loader, batch_size)  # validation 추가 , loss = val_3total_mean_loss
        
        wandb.log({"loss" : loss,
                   "Validation_3_total_loss" : val_3total_mean_loss,
                   "val_cls_loss" : cls_loss,
                   "val_angle_loss" : angle_loss,
                   "val_iou_loss" : iou_loss
                  })
        
        if val_3total_mean_loss < min_loss:  # 3가지 loss의 합이 최소일때 모델 저장
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'best_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            min_loss = val_3total_mean_loss
            
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            
        if (epoch + 1) == 45: # 100-55 = 45
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, 'epoch_100.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            
        if (epoch + 1) == 65: # 120 - 55 = 65
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            ckpt_fpath = osp.join(model_dir, 'epoch_120.pth')
            torch.save(model.state_dict(), ckpt_fpath)

            
def validation(model, val_loader, num_batches):
    model.eval()
    epoch_loss, epoch_start = 0, time.time()
    total_loss = 0
    cls_loss = 0
    angle_loss = 0
    iou_loss = 0
    with tqdm(total=len(val_loader)) as pbar:
        with torch.no_grad():
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                pbar.set_description('[Validation]')

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                loss_val = loss.item()
                epoch_loss += loss_val

                val_dict = {
                    'val Cls loss': extra_info['cls_loss'], 'val Angle loss': extra_info['angle_loss'],
                    'val IoU loss': extra_info['iou_loss'], 
                    'val Total loss': extra_info['cls_loss']+extra_info['angle_loss']+extra_info['iou_loss']
                }
                
                total_loss += val_dict['val Total loss']
                cls_loss += val_dict['val Cls loss']
                angle_loss += val_dict['val Angle loss']
                iou_loss += val_dict['val IoU loss']
                
                pbar.set_postfix(val_dict)
                
    return epoch_loss/len(val_loader), total_loss/len(val_loader), cls_loss/len(val_loader), angle_loss/len(val_loader), iou_loss/len(val_loader)
            
                
def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
