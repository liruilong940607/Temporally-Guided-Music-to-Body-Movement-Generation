import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import Download, audio_skeleton_dataset
from argument import parse
from metric import L1_loss
from model.utils import sort_sequences
from model.network import MovementNet
from model.optimizer import Optimizer
import vedo

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def main():
    parser = parse()
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_ids = [i for i in range(len(args.gpu_ids.split(',')))]

    # Data
    if args.aist:
        train_dataset = audio_skeleton_dataset(args.anno_dir, 'train', is_aist=True, smpl_dir=args.smpl_dir, audio_dir=args.audio_dir)
        val_dataset = audio_skeleton_dataset(args.anno_dir, 'valtest', is_aist=True, smpl_dir=args.smpl_dir, audio_dir=args.audio_dir)
    else:
        download_data = Download()
        download_data.train_data()
        train_dataset = audio_skeleton_dataset(download_data.train_dst, 'train')
        val_dataset = audio_skeleton_dataset(download_data.train_dst, 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    # Model
    movement_net = MovementNet(args.d_input, args.d_output_body, args.d_output_rh, args.d_model, args.n_block, args.n_unet, args.n_attn, args.n_head, args.max_len, args.dropout,
                                   args.pre_lnorm, args.attn_type).to('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() and len(args.gpu_ids.split(',')) > 1:
        movement_net = nn.DataParallel(movement_net, device_ids=gpu_ids)
    optimizer = Optimizer(
        torch.optim.Adam(movement_net.parameters(), betas=(0.9, 0.98), eps=1e-09),
        1.0,
        args.d_model,
        args.warmup_steps)

    #------------------------ START TRAINING ---------------------------------#
    print('Training... \n' )
    counter = 0
    min_val_loss = float('inf')

    Epoch_train_loss = []
    Epoch_val_loss = []
    for e in range(args.epoch):
        print("epoch %d" %(e+1))

        # Training stage
        movement_net.train()

        pose_loss = []
        for X_train, y_train, seq_len in train_loader:

            X_train, lengths = sort_sequences(X_train, seq_len)
            y_train, _ = sort_sequences(y_train, seq_len)
            mask = y_train != 0
            mask = mask.type('torch.FloatTensor').to('cuda:0' if torch.cuda.is_available() else 'cpu')

            full_output = movement_net.forward(X_train, lengths)

            loss = L1_loss(full_output, y_train, mask[:, :, :1])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(movement_net.parameters(), 1.)
            optimizer.step()

            pose_loss.append(loss.data.cpu().numpy())

        Epoch_train_loss.append(np.mean(pose_loss))
        print('train loss: ' + str(np.mean(pose_loss)))

        # Validation stage
        movement_net.eval()

        if args.aist and True: # visualize
            print ("starting visualize")
            pred = full_output[0].detach().cpu().numpy()
        
            # Transform keypoints to world coordinate
            pred = pred * train_dataset.keypoints_std + train_dataset.keypoints_mean
            pred = np.reshape(pred, [len(pred), -1, 3])
        
            for kpts in pred[:300]:
                pts = vedo.Points(kpts, r=20)
                vedo.show(pts, interactive=False)

        pose_loss = []
        with torch.no_grad():
            for X_val, y_val, seq_len in val_loader:

                X_val, lengths = sort_sequences(X_val, seq_len)
                y_val, _ = sort_sequences(y_val, seq_len)
                mask = y_val != 0
                mask = mask.type('torch.FloatTensor').to('cuda:0' if torch.cuda.is_available() else 'cpu')

                full_output = movement_net.forward(X_val, lengths)

                loss = L1_loss(full_output, y_val, mask[:, :, :1])
                pose_loss.append(loss.data.cpu().numpy())

            Epoch_val_loss.append(np.mean(pose_loss))
            print('val loss: ' + str(np.mean(pose_loss)) + '\n')

            if counter == args.early_stop_iter:
                print("------------------early stopping------------------\n")
                break
            else:
                if min_val_loss > np.mean(pose_loss):
                    min_val_loss = np.mean(pose_loss)
                    counter = 0
                    if not os.path.exists('checkpoint'):
                        os.makedirs('checkpoint')
                    if torch.cuda.is_available() and len(args.gpu_ids.split(',')) > 1:
                        state_dict = movement_net.module.state_dict()
                    else:
                        state_dict = movement_net.state_dict()
                    torch.save({'epoch' : e+1,
                                'model_state_dict': {'movement_net': state_dict},
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': min_val_loss}, args.checkpoint)
                else:
                    counter += 1

if __name__ == '__main__':
    main()