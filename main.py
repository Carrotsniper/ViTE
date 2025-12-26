import os
import argparse
from csv import writer
import torch
import numpy as np
from utils.helper import *
from model.ViTE_model import ViTE
from utils.dataloader_eth import TrajectoryDataset

def main():
    data_root = os.path.join('./datasets/ethucy', args.dataset)
    dataset_test = TrajectoryDataset(args, os.path.join(data_root, 'test'), obs_len=opts.past_length, pred_len=opts.future_length, skip=1)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)
    model = ViTE(opts).cuda()
    checkpoint_name = args.model_name if args.model_name else os.path.basename(args.config).split('.')[0]
    model_save_dir = os.path.join('./checkpoint', checkpoint_name)
    os.makedirs(model_save_dir, exist_ok=True)

    if args.test:
        model_name = args.dataset + '_ckpt_best.pth'
        model_path = os.path.join(model_save_dir, model_name)
        print('[INFO] Loading model from:', model_path)
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt['state_dict'], strict=True)
        ade, fde = test(model_ckpt['epoch'], model, loader_test)
        os.makedirs('results', exist_ok=True)
        with open(os.path.join('./results', '{}_result.csv'.format(args.dataset)), 'w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerow([os.path.basename(args.config).split('.')[0], ade, fde])
        exit()


def test(epoch, model, loader):
    model.eval()
    avg_meter = {'epoch': epoch, 'ade': 0, 'fde': 0, 'counter': 0}
    
    with torch.no_grad():
        for _, data in enumerate(loader):
            x_abs, y = data
            x_abs, y = x_abs.cuda(), y.cuda()        
            
            batch_size, num_agents, length, _ = x_abs.size()

            x_rel = torch.zeros_like(x_abs)
            x_rel[:, :, 1:] = x_abs[:, :, 1:] - x_abs[:, :, :-1]
            x_rel[:, :, 0] = x_rel[:, :, 1] 
            
            y_pred, _ = model(x_abs, x_rel, log_moe_info=False)

            if opts.pred_rel:
                cur_pos = x_abs[:, :, [-1]].unsqueeze(2)
                y_pred = torch.cumsum(y_pred, dim=3) + cur_pos

            y_pred = np.array(y_pred.cpu())
            y = np.array(y.cpu())
            y = y[:, :, None, :, :]
            
            ade = np.mean(np.min(np.mean(np.linalg.norm(y_pred - y, axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
            fde = np.mean(np.min(np.mean(np.linalg.norm(y_pred[:, :, :, -1:] - y[:, :, :, -1:], axis=-1), axis=3), axis=2)) * (num_agents * batch_size)
                        
            avg_meter['ade'] += ade
            avg_meter['fde'] += fde
            
            avg_meter['counter'] += (num_agents * batch_size)
    
    print('\n[{}][{}] Epoch {}'.format(args.dataset.upper(), 'TEST', epoch))
    print('[{}][{}] minADE/minFDE: {:.4f}/{:.4f}'.format(args.dataset.upper(), 'TEST', avg_meter['ade'] / avg_meter['counter'], avg_meter['fde'] / avg_meter['counter']))
    
    return avg_meter['fde'] / avg_meter['counter'], avg_meter['ade'] / avg_meter['counter']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViTE for Trajectory Prediction')
    parser.add_argument('--dataset', type=str, default='univ', metavar='N', help='dataset name')
    parser.add_argument('--config', type=str, default='./configs/ethucy.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument("--test", action='store_true')
    parser.add_argument('--model_name', type=str, default='univ', help='model name for checkpoint saving')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    opts = load_config(args.config)
    main()