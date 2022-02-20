# network wrappers for 3D-RGAN
# Updated: 12/25/2021
# Status: OK

import torch
import os, sys
import glob
import shutil
import wandb

import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from scipy import interpolate

from dataloader import B_Data, ParcellationDataBinary
from models import _G_Model, _D_Model, _Gs_Model, _CNN, _MLP_Surv
from utils import write_raw_score
from loss_functions import sur_loss

torch.set_num_threads(1) # may lower training speed (and potentially overheat issue) if machine has many cpus!
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class RGAN_Wrapper:
    def __init__(self, config, model_name, SWEEP=False):
        self.SWEEP = SWEEP
        self.config = config
        self.data_dir = config['data_dir']
        self.lr_d = config['lr_d']
        self.lr_g = config['lr_g']
        self.seed = 1000
        self.model_name = model_name
        self.loss_metric = config['loss_metric']
        self.checkpoint_dir = './checkpoint_dir/'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += '{}/'.format(self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir+'output_dir/'
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        torch.manual_seed(self.seed)
        self.prepare_dataloader(config['batch_size'], self.data_dir)

        # in_size = 121*145*121
        self.g = _G_Model(config).to(device)
        self.d = _D_Model(config).to(device)

        if self.loss_metric == 'Standard':
            self.criterion = nn.BCELoss().to(device)
        else:
            # self.criterion = nn.CrossEntropyLoss().to(device)
            self.criterion = nn.MSELoss().to(device)

        if config['optimizer'] == 'SGD':
            self.optimizer_g = optim.SGD(self.g.parameters(), lr=config['lr_g'], weight_decay=config['weight_decay_g'])
            self.optimizer_d = optim.SGD(self.d.parameters(), lr=config['lr_d'], weight_decay=config['weight_decay_d'])
        elif config['optimizer'] == 'Adam':
            self.optimizer_g = optim.Adam(self.g.parameters(), lr=config['lr_g'], weight_decay=config['weight_decay_g'])
            self.optimizer_d = optim.Adam(self.d.parameters(), lr=config['lr_d'], weight_decay=config['weight_decay_d'])

    def prepare_dataloader(self, batch_size, data_dir):
        self.train_data = B_Data(data_dir, stage='train', seed=self.seed, step_size=self.config['step_size'])
        sample_weight, self.imbalanced_ratio = self.train_data.get_sample_weights()
        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False, drop_last=True)

        valid_data = B_Data(data_dir, stage='valid', seed=self.seed, step_size=self.config['step_size'])
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

        self.test_data  = B_Data(data_dir, stage='test', seed=self.seed, step_size=self.config['step_size'])
        self.test_dataloader = DataLoader(self.test_data, batch_size=1, shuffle=False)

        self.all_data = B_Data(data_dir, stage='all', seed=self.seed, step_size=self.config['step_size'])
        self.all_dataloader = DataLoader(self.all_data, batch_size=1)

        Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
        external_data = B_Data(Data_dir_NACC, stage='all', seed=self.seed, step_size=self.config['step_size'], external=True)
        self.external_data = external_data
        self.ext_dataloader = DataLoader(self.external_data, batch_size=1)

    def load(self, dir=None, fixed=False):
        if dir:
            # need to update
            print('loading pre-trained model...')
            dir = [glob.glob(dir[0] + '*_d_*.pth')[0], glob.glob(dir[1] + '*_g_*.pth')[0]]
            print('might need update')
        else:
            print('loading model...')
            dir = [glob.glob(self.checkpoint_dir + '*_d_*.pth')[0], glob.glob(self.checkpoint_dir + '*_g_*.pth')[0]]
        st_d = torch.load(dir[0])
        st_g = torch.load(dir[1])
        # del st['l2.weight']
        self.d.load_state_dict(st_d, strict=False)
        self.g.load_state_dict(st_g, strict=False)
        if fixed:
            print('need update')
            sys.exit()
            ps = []
            for n, p in self.model.named_parameters():
                if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # for n, p in self.model.named_parameters():
            # print(n, p.requires_grad)
        print('loaded.')

    def train(self, epochs):
        print('training...')
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        self.epoch = 0
        self.generate([self.valid_dataloader])

        d_gs, d_rs = [], []
        for self.epoch in range(1, epochs+1):
            train_loss = self.train_model_epoch()
            d_loss, g_loss, d_g, d_g2, d_r = train_loss
            d_gs += [d_g]
            d_rs += [d_r]
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                self.generate([self.valid_dataloader])
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)

                end.record()
                torch.cuda.synchronize()
                print('{}th epoch g_valid loss [{}] ='.format(self.epoch, self.config['loss_metric']), '%.3f' % (val_loss), 'Loss_D: %.3f Loss_G: %.3f D(x): %.3f D(G(z)): %.3f / %.3f' % (d_loss, g_loss, d_r, d_g, d_g2), '|| time(s) =', int(start.elapsed_time(end)//1000)) #d_g - d_g2: before update vs after update for d
                if self.SWEEP:
                    wandb.log({"g_valid_loss":val_loss})
                    wandb.log({"D loss":d_loss})
                    wandb.log({"G loss":g_loss})
                    wandb.log({"D(x)":d_r})
                    wandb.log({"D(G(z))":d_g})

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        self.plot_train(d_rs, d_gs)
        return self.optimal_valid_metric

    def plot_train(self, d_rs, d_gs):
        plt.figure(figsize=(10,5))
        plt.title('Training Loss')
        plt.plot(d_rs, label='R')
        plt.plot(d_gs, label='G')
        plt.xlabel("epochs")
        plt.ylabel("preds")
        plt.legend()
        # plt.show()
        plt.savefig(self.output_dir+'train_loss.png', dpi=150)
        plt.close()

    def train_model_epoch(self):
        self.g.train(True)
        self.d.train(True)
        d_loss, g_loss = [], []
        d_g, d_g2, d_r = [], [], []

        torch.use_deterministic_algorithms(False)
        for inputs, targets, _, _ in self.train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            self.g.zero_grad()
            self.d.zero_grad()

            #update d
            r_preds = self.d(targets).view(-1)
            r_labels = torch.full((r_preds.shape[0],), 1, dtype=torch.float, device=device)
            loss_r = self.criterion(r_preds, r_labels)

            loss_r.backward()
            d_r += [r_preds.mean().item()]

            g_out = self.g(inputs)
            g_preds = self.d(g_out.detach()).view(-1)#don't want to update g here
            g_labels = torch.full((g_preds.shape[0],), 0, dtype=torch.float, device=device)
            loss_g = self.criterion(g_preds, g_labels)
            loss_g.backward()
            d_g += [g_preds.mean().item()]
            loss_d = loss_g+loss_r
            d_loss += [loss_d.item()]
            if self.epoch % 5 == 0:
                self.optimizer_d.step()

            #update g
            g_out = self.g(inputs)
            preds = self.d(g_out).view(-1)#want to update g here
            labels = torch.full((preds.shape[0],), 1, dtype=torch.float, device=device) #target for g should be 1
            p_loss = torch.mean(torch.abs(g_out-targets))
            loss = self.criterion(preds, labels)+self.config['p_weight']*p_loss
            loss.backward()
            g_loss += [loss.item()]
            d_g2 += [preds.mean().item()]
            self.optimizer_g.step()

            # print(D_g, D_g2)
            # sys.exit()
            # clip = 1
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        torch.use_deterministic_algorithms(True)
        # print((d_loss), (g_loss), (d_g), (d_g2))
        return np.mean(d_loss), np.mean(g_loss), np.mean(d_g), np.mean(d_g2), np.mean(d_r)

    def valid_model_epoch(self):
        self.g.eval()
        self.d.eval()
        with torch.no_grad():
            loss_all = []
            for inputs, targets, _, _ in self.valid_dataloader:
                # here only use 1 patch
                inputs, targets = inputs.to(device), targets.to(device)
                g_out = self.g(inputs)
                g_preds = self.d(g_out).view(-1)
                g_labels = torch.full((g_preds.shape[0],), 1, dtype=torch.float, device=device)
                loss_g = self.criterion(g_preds, g_labels)
                loss_all += [loss_g.item()]
        return np.mean(loss_all)*5

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score

            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.d.state_dict(), '{}{}_d_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.g.state_dict(), '{}{}_g_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def generate(self, datas, whole=False, samples=True, ext=False):
        self.g.eval()
        if whole:
            #prepare locations for data generation!
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_dirs = [out_dir + 'Z/', out_dir + 'G/', out_dir + 'T/']
            if ext:
                out_dirs += [out_dir + 'Z_E/', out_dir + 'G_E/', out_dir + 'T_E/']
            for o in out_dirs:
                if os.path.isdir(o):
                    shutil.rmtree(o)
                if not os.path.exists(o):
                    os.mkdir(o)

        with torch.no_grad():
            for idx, data in enumerate(datas):
                for inputs, targets, fname, _ in data:
                    # here only use 1 patch
                    inputs, targets = inputs.to(device), targets.to(device)
                    fname = os.path.basename(fname[0])
                    g_out = self.g(inputs).cpu().numpy().squeeze()
                    targets = targets.cpu().numpy().squeeze()

                    inputs = inputs.cpu().numpy().squeeze()
                    out = np.zeros(targets.shape)
                    out[::self.config['step_size']] = inputs

                    if samples:
                        dir_name = self.output_dir+str(self.epoch)+'_'
                        dir_name += fname
                        self.visualize(out, g_out, targets, dir_name)
                    if whole:
                        # generate for all data
                        img_z = nib.Nifti1Image(out, np.eye(4))
                        img_z.to_filename(out_dirs[0+3*idx]+fname)

                        img_g = nib.Nifti1Image(g_out, np.eye(4))
                        img_g.to_filename(out_dirs[1+3*idx]+fname)

                        img_t = nib.Nifti1Image(targets, np.eye(4))
                        img_t.to_filename(out_dirs[2+3*idx]+fname)
                    else:
                        break

    def visualize(self, src, gen, tar, dir):
        plt.set_cmap("gray")
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig, axs = plt.subplots(3, 9, figsize=(20, 15))

        step = np.array(gen.shape)//3
        offset = step//2

        for i in range(3):

            axs[i, 0].imshow(src[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 0].set_title('Z: x_{}'.format(i), fontsize=25)
            axs[i, 0].axis('off')
            axs[i, 1].imshow(gen[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 1].set_title('G: x_{}'.format(i), fontsize=25)
            axs[i, 1].axis('off')
            axs[i, 2].imshow(tar[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 2].set_title('T: x_{}'.format(i), fontsize=25)
            axs[i, 2].axis('off')

            axs[i, 3].imshow(src[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 3].set_title('Z: y_{}'.format(i), fontsize=25)
            axs[i, 3].axis('off')
            axs[i, 4].imshow(gen[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 4].set_title('G: y_{}'.format(i), fontsize=25)
            axs[i, 4].axis('off')
            axs[i, 5].imshow(tar[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 5].set_title('T: y_{}'.format(i), fontsize=25)
            axs[i, 5].axis('off')

            axs[i, 6].imshow(src[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 6].set_title('Z: z_{}'.format(i), fontsize=25)
            axs[i, 6].axis('off')
            axs[i, 7].imshow(gen[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 7].set_title('G: z_{}'.format(i), fontsize=25)
            axs[i, 7].axis('off')
            axs[i, 8].imshow(tar[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 8].set_title('T: z_{}'.format(i), fontsize=25)
            axs[i, 8].axis('off')
        plt.savefig(dir.replace('nii', 'png'), dpi=150)
        plt.close()
        '''
        def bold_axs_stick(axs, fontsize):
            for tick in axs.xaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)
                tick.label1.set_fontweight('bold')
            for tick in axs.yaxis.get_major_ticks():
                tick.label1.set_fontsize(fontsize)
                tick.label1.set_fontweight('bold')
        axs[0, 2].hist(img15[side_a:side_b, side_a:side_b, 105].T.flatten(), bins=50, range=(0, 1.8))
        bold_axs_stick(axs[0, 2], 16)
        axs[0, 2].set_xticks([0, 0.5, 1, 1.5])
        axs[0, 2].set_yticks([0, 100, 200, 300])
        axs[0, 2].set_title('1.5T voxel histogram', fontsize=25)
        axs[1, 2].hist(img3[side_a:side_b, side_a:side_b, 105].T.flatten(), bins=50, range=(0, 1.8))
        bold_axs_stick(axs[1, 2], 16)
        axs[1, 2].set_xticks([0, 0.5, 1, 1.5])
        axs[1, 2].set_yticks([0, 100, 200, 300])
        axs[1, 2].set_title('3T voxel histogram', fontsize=25)
        axs[2, 2].hist(imgp[side_a:side_b, side_a:side_b, 105].T.flatten(), bins=50, range=(0, 1.8))
        bold_axs_stick(axs[1, 2], 16)
        axs[2, 2].set_xticks([0, 0.5, 1, 1.5])
        axs[2, 2].set_yticks([0, 100, 200, 300])
        axs[2, 2].set_title('1.5T+ voxel histogram', fontsize=25)
        '''


class RCGAN_Wrapper:
    def __init__(self, config, model_name, SWEEP=False):
        self.SWEEP = SWEEP
        self.config = config
        self.data_dir = config['data_dir']
        self.lr_d = config['lr_d']
        self.lr_g = config['lr_g']
        self.seed = 1000
        self.model_name = model_name
        self.loss_metric = config['loss_metric']
        self.checkpoint_dir = './checkpoint_dir/'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += '{}/'.format(self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir+'output_dir/'
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        torch.manual_seed(self.seed)
        self.prepare_dataloader(config['batch_size'], self.data_dir)

        # in_size = 121*145*121
        self.g = _G_Model(config).to(device)
        self.d = _D_Model(config).to(device)
        self.c = _CNN(config).to(device)

        if self.loss_metric == 'Standard':
            self.criterion = nn.BCELoss().to(device)
        else:
            # self.criterion = nn.CrossEntropyLoss().to(device)
            self.criterion = nn.MSELoss().to(device)

        if config['optimizer'] == 'SGD':
            self.optimizer_g = optim.SGD(self.g.parameters(), lr=config['lr_g'], weight_decay=config['weight_decay_g'])
            self.optimizer_d = optim.SGD(self.d.parameters(), lr=config['lr_d'], weight_decay=config['weight_decay_d'])
            self.optimizer_c = optim.SGD(self.c.parameters(), lr=config['lr_c'], weight_decay=config['weight_decay_c'])
        elif config['optimizer'] == 'Adam':
            self.optimizer_g = optim.Adam(self.g.parameters(), lr=config['lr_g'], weight_decay=config['weight_decay_g'])
            self.optimizer_d = optim.Adam(self.d.parameters(), lr=config['lr_d'], weight_decay=config['weight_decay_d'])
            self.optimizer_c = optim.Adam(self.c.parameters(), lr=config['lr_c'], weight_decay=config['weight_decay_c'])

    def prepare_dataloader(self, batch_size, data_dir):
        self.train_data = B_Data(data_dir, stage='train', seed=self.seed, step_size=self.config['step_size'])
        sample_weight, self.imbalanced_ratio = self.train_data.get_sample_weights()
        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False, drop_last=True)

        valid_data = B_Data(data_dir, stage='valid', seed=self.seed, step_size=self.config['step_size'])
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

        self.test_data  = B_Data(data_dir, stage='test', seed=self.seed, step_size=self.config['step_size'])
        self.test_dataloader = DataLoader(self.test_data, batch_size=1, shuffle=False)

        self.all_data = B_Data(data_dir, stage='all', seed=self.seed, step_size=self.config['step_size'])
        self.all_dataloader = DataLoader(self.all_data, batch_size=1)

        Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
        external_data = B_Data(Data_dir_NACC, stage='all', seed=self.seed, step_size=self.config['step_size'], external=True)
        self.external_data = external_data
        self.ext_dataloader = DataLoader(self.external_data, batch_size=1)

    def load(self, dir=None, fixed=False):
        if dir:
            # need to update
            print('loading pre-trained model...')
            dir = [glob.glob(dir[0] + '*_d_*.pth')[0], glob.glob(dir[1] + '*_g_*.pth')[0], glob.glob(dir[2] + '*_c_*.pth')[0]]
            print('might need update')
        else:
            print('loading model...')
            dir = [glob.glob(self.checkpoint_dir + '*_d_*.pth')[0], glob.glob(self.checkpoint_dir + '*_g_*.pth')[0], glob.glob(self.checkpoint_dir + '*_c_*.pth')[0]]
        st_d = torch.load(dir[0])
        st_g = torch.load(dir[1])
        st_c = torch.load(dir[2])
        # del st['l2.weight']
        self.d.load_state_dict(st_d, strict=False)
        self.g.load_state_dict(st_g, strict=False)
        self.c.load_state_dict(st_c, strict=False)
        if fixed:
            print('need update')
            sys.exit()
            ps = []
            for n, p in self.model.named_parameters():
                if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # for n, p in self.model.named_parameters():
            # print(n, p.requires_grad)
        print('loaded.')

    def train(self, epochs):
        print('training...')
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        self.epoch = 0
        self.generate([self.valid_dataloader])

        d_gs, d_rs = [], []
        for self.epoch in range(1, epochs+1):
            train_loss = self.train_model_epoch()
            d_loss, g_loss, c_loss, d_g, d_g2, d_r = train_loss
            d_gs += [d_g]
            d_rs += [d_r]
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                self.generate([self.valid_dataloader])
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)

                end.record()
                torch.cuda.synchronize()
                print('{}th epoch g_valid loss [{}] ='.format(self.epoch, self.config['loss_metric']), '%.3f' % (val_loss), 'Loss_C: %.3f Loss_D: %.3f Loss_G: %.3f D(x): %.3f D(G(z)): %.3f / %.3f' % (c_loss, d_loss, g_loss, d_r, d_g, d_g2), '|| time(s) =', int(start.elapsed_time(end)//1000)) #d_g - d_g2: before update vs after update for d
                if self.SWEEP:
                    wandb.log({"g_valid_loss":val_loss})
                    wandb.log({"C_loss":c_loss})
                    wandb.log({"D loss":d_loss})
                    wandb.log({"G loss":g_loss})
                    wandb.log({"D(x)":d_r})
                    wandb.log({"D(G(z))":d_g})

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        self.plot_train(d_rs, d_gs)
        return self.optimal_valid_metric

    def plot_train(self, d_rs, d_gs):
        plt.figure(figsize=(10,5))
        plt.title('Training Loss')
        plt.plot(d_rs, label='R')
        plt.plot(d_gs, label='G')
        plt.xlabel("epochs")
        plt.ylabel("preds")
        plt.legend()
        # plt.show()
        plt.savefig(self.output_dir+'train_loss.png', dpi=150)
        plt.close()

    def train_model_epoch(self):
        self.g.train(True)
        self.d.train(True)
        self.c.train(True)
        d_losses, g_losses, c_losses = [], [], []
        d_g, d_g2, d_r = [], [], []

        torch.use_deterministic_algorithms(False)
        for inputs, targets, _, labels in self.train_dataloader:
            inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device).float()
            self.g.zero_grad()
            self.d.zero_grad()
            self.c.zero_grad()

            #update d
            r_preds = self.d(targets).view(-1)
            r_labels = torch.full((r_preds.shape[0],), 1, dtype=torch.float, device=device)
            loss_r = self.criterion(r_preds, r_labels)

            loss_r.backward()
            d_r += [r_preds.mean().item()]

            g_out = self.g(inputs)
            g_preds = self.d(g_out.detach()).view(-1)#don't want to update g here
            g_labels = torch.full((g_preds.shape[0],), 0, dtype=torch.float, device=device)
            loss_g = self.criterion(g_preds, g_labels)
            loss_g.backward()
            d_g += [g_preds.mean().item()]
            loss_d = loss_g+loss_r
            d_losses += [loss_d.item()]
            if self.epoch % 5 == 0:
                self.optimizer_d.step()

            #update g and c
            g_out = self.g(inputs)
            preds_c = self.c(g_out).view(-1)
            c_loss = self.criterion(preds_c, labels)
            c_losses += [c_loss.item()]

            preds_d = self.d(g_out).view(-1)#want to update g here
            labels_d = torch.full((preds_d.shape[0],), 1, dtype=torch.float, device=device) #target for g should be 1
            d_loss = self.criterion(preds_d, labels_d)

            p_loss = torch.mean(torch.abs(g_out-targets))

            loss = self.config['c_weight']*c_loss + self.config['d_weight']*d_loss + self.config['p_weight']*p_loss
            loss.backward()
            g_losses += [loss.item()]
            d_g2 += [preds_d.mean().item()]
            self.optimizer_g.step()
            self.optimizer_c.step()

            # print(D_g, D_g2)
            # sys.exit()
            # clip = 1
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        torch.use_deterministic_algorithms(True)
        # print((d_loss), (g_loss), (d_g), (d_g2))
        return np.mean(d_losses), np.mean(g_losses), np.mean(c_losses), np.mean(d_g), np.mean(d_g2), np.mean(d_r)

    def valid_model_epoch(self):
        self.g.eval()
        self.d.eval()
        self.c.eval()
        with torch.no_grad():
            loss_all = []
            for inputs, targets, _, labels in self.valid_dataloader:
                # here only use 1 patch
                inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device).float()

                g_out = self.g(inputs)
                preds_c = self.c(g_out).view(-1)
                c_loss = self.criterion(preds_c, labels)

                preds_d = self.d(g_out).view(-1)
                labels_d = torch.full((preds_d.shape[0],), 1, dtype=torch.float, device=device) #target for g should be 1
                d_loss = self.criterion(preds_d, labels_d)

                p_loss = torch.mean(torch.abs(g_out-targets))

                loss = self.config['c_weight']*c_loss + self.config['d_weight']*d_loss + self.config['p_weight']*p_loss
                loss_all += [loss.item()]

        return np.mean(loss_all)*self.config['batch_size']

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score

            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.d.state_dict(), '{}{}_d_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.g.state_dict(), '{}{}_g_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.c.state_dict(), '{}{}_c_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def generate(self, datas, whole=False, samples=True, ext=False):
        self.g.eval()

        if whole:
            #prepare locations for data generation!
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_dirs = [out_dir + 'Z/', out_dir + 'CG_1/', out_dir + 'T/']
            if ext:
                out_dirs += [out_dir + 'Z_E/', out_dir + 'CG_1_E/', out_dir + 'T_E/']
            for o in out_dirs:
                if os.path.isdir(o):
                    shutil.rmtree(o)
                if not os.path.exists(o):
                    os.mkdir(o)

        with torch.no_grad():
            for idx, data in enumerate(datas):
                for inputs, targets, fname, _ in data:
                    # here only use 1 patch
                    inputs, targets = inputs.to(device), targets.to(device)
                    fname = os.path.basename(fname[0])
                    g_out = self.g(inputs).cpu().numpy().squeeze()
                    targets = targets.cpu().numpy().squeeze()

                    inputs = inputs.cpu().numpy().squeeze()
                    out = np.zeros(targets.shape)
                    out[::self.config['step_size']] = inputs

                    if samples:
                        dir_name = self.output_dir+str(self.epoch)+'_'
                        dir_name += fname
                        self.visualize(out, g_out, targets, dir_name)
                    if whole:
                        # generate for all data
                        img_z = nib.Nifti1Image(out, np.eye(4))
                        img_z.to_filename(out_dirs[0+3*idx]+fname)

                        img_g = nib.Nifti1Image(g_out, np.eye(4))
                        img_g.to_filename(out_dirs[1+3*idx]+fname)

                        img_t = nib.Nifti1Image(targets, np.eye(4))
                        img_t.to_filename(out_dirs[2+3*idx]+fname)
                    else:
                        break

    def visualize(self, src, gen, tar, dir):
        plt.set_cmap("gray")
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig, axs = plt.subplots(3, 9, figsize=(20, 15))

        step = np.array(gen.shape)//3
        offset = step//2

        for i in range(3):

            axs[i, 0].imshow(src[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 0].set_title('Z: x_{}'.format(i), fontsize=25)
            axs[i, 0].axis('off')
            axs[i, 1].imshow(gen[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 1].set_title('G: x_{}'.format(i), fontsize=25)
            axs[i, 1].axis('off')
            axs[i, 2].imshow(tar[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 2].set_title('T: x_{}'.format(i), fontsize=25)
            axs[i, 2].axis('off')

            axs[i, 3].imshow(src[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 3].set_title('Z: y_{}'.format(i), fontsize=25)
            axs[i, 3].axis('off')
            axs[i, 4].imshow(gen[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 4].set_title('G: y_{}'.format(i), fontsize=25)
            axs[i, 4].axis('off')
            axs[i, 5].imshow(tar[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 5].set_title('T: y_{}'.format(i), fontsize=25)
            axs[i, 5].axis('off')

            axs[i, 6].imshow(src[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 6].set_title('Z: z_{}'.format(i), fontsize=25)
            axs[i, 6].axis('off')
            axs[i, 7].imshow(gen[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 7].set_title('G: z_{}'.format(i), fontsize=25)
            axs[i, 7].axis('off')
            axs[i, 8].imshow(tar[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 8].set_title('T: z_{}'.format(i), fontsize=25)
            axs[i, 8].axis('off')
        plt.savefig(dir.replace('nii', 'png'), dpi=150)
        plt.close()


class RCGANs_Wrapper:
    def __init__(self, config, model_name, SWEEP=False):
        self.SWEEP = SWEEP
        self.config = config
        self.data_dir = config['data_dir']
        self.lr_d = config['lr_d']
        self.lr_g = config['lr_g']
        self.seed = 1000
        self.model_name = model_name
        self.loss_metric = config['loss_metric']
        self.checkpoint_dir = './checkpoint_dir/'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += '{}/'.format(self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir+'output_dir/'
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        torch.manual_seed(self.seed)
        self.prepare_dataloader(config['batch_size'], self.data_dir)

        # in_size = 121*145*121
        self.g = _Gs_Model(config).to(device)
        self.d = _D_Model(config).to(device)
        self.c = _CNN(config).to(device)

        if self.loss_metric == 'Standard':
            self.criterion = nn.BCELoss().to(device)
        else:
            # self.criterion = nn.CrossEntropyLoss().to(device)
            self.criterion = nn.MSELoss().to(device)

        if config['optimizer'] == 'SGD':
            self.optimizer_g = optim.SGD(self.g.parameters(), lr=config['lr_g'], weight_decay=config['weight_decay_g'])
            self.optimizer_d = optim.SGD(self.d.parameters(), lr=config['lr_d'], weight_decay=config['weight_decay_d'])
            self.optimizer_c = optim.SGD(self.c.parameters(), lr=config['lr_c'], weight_decay=config['weight_decay_c'])
        elif config['optimizer'] == 'Adam':
            self.optimizer_g = optim.Adam(self.g.parameters(), lr=config['lr_g'], weight_decay=config['weight_decay_g'])
            self.optimizer_d = optim.Adam(self.d.parameters(), lr=config['lr_d'], weight_decay=config['weight_decay_d'])
            self.optimizer_c = optim.Adam(self.c.parameters(), lr=config['lr_c'], weight_decay=config['weight_decay_c'])

    def prepare_dataloader(self, batch_size, data_dir):
        self.train_data = B_Data(data_dir, stage='train', seed=self.seed, step_size=self.config['step_size'])
        sample_weight, self.imbalanced_ratio = self.train_data.get_sample_weights()
        self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=False, drop_last=True)

        valid_data = B_Data(data_dir, stage='valid', seed=self.seed, step_size=self.config['step_size'])
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

        self.test_data  = B_Data(data_dir, stage='test', seed=self.seed, step_size=self.config['step_size'])
        self.test_dataloader = DataLoader(self.test_data, batch_size=1, shuffle=False)

        self.all_data = B_Data(data_dir, stage='all', seed=self.seed, step_size=self.config['step_size'])
        self.all_dataloader = DataLoader(self.all_data, batch_size=1)

        Data_dir_NACC = "/data2/MRI_PET_DATA/processed_images_final_cox_test/brain_stripped_cox_test/"
        external_data = B_Data(Data_dir_NACC, stage='all', seed=self.seed, step_size=self.config['step_size'], external=True)
        self.external_data = external_data
        self.ext_dataloader = DataLoader(self.external_data, batch_size=1)

    def load(self, dir=None, fixed=False):
        if dir:
            # need to update
            print('loading pre-trained model...')
            dir = [glob.glob(dir[0] + '*_d_*.pth')[0], glob.glob(dir[1] + '*_g_*.pth')[0], glob.glob(dir[1] + '*_c_*.pth')[0]]
            print('might need update')
        else:
            print('loading model...')
            dir = [glob.glob(self.checkpoint_dir + '*_d_*.pth')[0], glob.glob(self.checkpoint_dir + '*_g_*.pth')[0], glob.glob(self.checkpoint_dir + '*_c_*.pth')[0]]
        st_d = torch.load(dir[0])
        st_g = torch.load(dir[1])
        st_c = torch.load(dir[2])
        # del st['l2.weight']
        self.d.load_state_dict(st_d, strict=False)
        self.g.load_state_dict(st_g, strict=False)
        self.c.load_state_dict(st_c, strict=False)
        if fixed:
            print('need update')
            sys.exit()
            ps = []
            for n, p in self.model.named_parameters():
                if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # for n, p in self.model.named_parameters():
            # print(n, p.requires_grad)
        print('loaded.')

    def train(self, epochs):
        print('training...')
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        self.epoch = 0
        self.generate([self.valid_dataloader])

        d_gs, d_rs = [], []
        for self.epoch in range(1, epochs+1):
            train_loss = self.train_model_epoch()
            d_loss, g_loss, c_loss, d_g, d_g2, d_r = train_loss
            d_gs += [d_g]
            d_rs += [d_r]
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                self.generate([self.valid_dataloader])
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)

                end.record()
                torch.cuda.synchronize()
                print('{}th epoch g_valid loss [{}] ='.format(self.epoch, self.config['loss_metric']), '%.3f' % (val_loss), 'Loss_C: %.3f Loss_D: %.3f Loss_G: %.3f D(x): %.3f D(G(z)): %.3f / %.3f' % (c_loss, d_loss, g_loss, d_r, d_g, d_g2), '|| time(s) =', int(start.elapsed_time(end)//1000)) #d_g - d_g2: before update vs after update for d
                if self.SWEEP:
                    wandb.log({"g_valid_loss":val_loss})
                    wandb.log({"C_loss":c_loss})
                    wandb.log({"D loss":d_loss})
                    wandb.log({"G loss":g_loss})
                    wandb.log({"D(x)":d_r})
                    wandb.log({"D(G(z))":d_g})

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        self.plot_train(d_rs, d_gs)
        return self.optimal_valid_metric

    def plot_train(self, d_rs, d_gs):
        plt.figure(figsize=(10,5))
        plt.title('Training Loss')
        plt.plot(d_rs, label='R')
        plt.plot(d_gs, label='G')
        plt.xlabel("epochs")
        plt.ylabel("preds")
        plt.legend()
        # plt.show()
        plt.savefig(self.output_dir+'train_loss.png', dpi=150)
        plt.close()

    def train_model_epoch(self):
        self.g.train(True)
        self.d.train(True)
        self.c.train(True)
        d_losses, g_losses, c_losses = [], [], []
        d_g, d_g2, d_r = [], [], []

        torch.use_deterministic_algorithms(False)
        for inputs, targets, _, labels in self.train_dataloader:
            inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device).float()
            self.g.zero_grad()
            self.d.zero_grad()
            self.c.zero_grad()

            #update d
            r_preds = self.d(targets).view(-1)
            r_labels = torch.full((r_preds.shape[0],), 1, dtype=torch.float, device=device)
            loss_r = self.criterion(r_preds, r_labels)

            loss_r.backward()
            d_r += [r_preds.mean().item()]

            g_out = self.g(inputs)
            g_preds = self.d(g_out.detach()).view(-1)#don't want to update g here
            g_labels = torch.full((g_preds.shape[0],), 0, dtype=torch.float, device=device)
            loss_g = self.criterion(g_preds, g_labels)
            loss_g.backward()
            d_g += [g_preds.mean().item()]
            loss_d = loss_g+loss_r
            d_losses += [loss_d.item()]
            if self.epoch % 5 == 0:
                self.optimizer_d.step()

            #update g and c
            g_out = self.g(inputs)
            preds_c = self.c(g_out).view(-1)
            c_loss = self.criterion(preds_c, labels)
            c_losses += [c_loss.item()]

            preds_d = self.d(g_out).view(-1)#want to update g here
            labels_d = torch.full((preds_d.shape[0],), 1, dtype=torch.float, device=device) #target for g should be 1
            d_loss = self.criterion(preds_d, labels_d)

            p_loss = torch.mean(torch.abs(g_out-targets))

            loss = self.config['c_weight']*c_loss + self.config['d_weight']*d_loss + self.config['p_weight']*p_loss
            loss.backward()
            g_losses += [loss.item()]
            d_g2 += [preds_d.mean().item()]
            self.optimizer_g.step()
            self.optimizer_c.step()

            # print(D_g, D_g2)
            # sys.exit()
            # clip = 1
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        torch.use_deterministic_algorithms(True)
        # print((d_loss), (g_loss), (d_g), (d_g2))
        return np.mean(d_losses), np.mean(g_losses), np.mean(c_losses), np.mean(d_g), np.mean(d_g2), np.mean(d_r)

    def valid_model_epoch(self):
        self.g.eval()
        self.d.eval()
        self.c.eval()
        with torch.no_grad():
            loss_all = []
            for inputs, targets, _, labels in self.valid_dataloader:
                # here only use 1 patch
                inputs, targets, labels = inputs.to(device), targets.to(device), labels.to(device).float()

                g_out = self.g(inputs)
                preds_c = self.c(g_out).view(-1)
                c_loss = self.criterion(preds_c, labels)

                preds_d = self.d(g_out).view(-1)
                labels_d = torch.full((preds_d.shape[0],), 1, dtype=torch.float, device=device) #target for g should be 1
                d_loss = self.criterion(preds_d, labels_d)

                p_loss = torch.mean(torch.abs(g_out-targets))

                loss = self.config['c_weight']*c_loss + self.config['d_weight']*d_loss + self.config['p_weight']*p_loss
                loss_all += [loss.item()]

        return np.mean(loss_all)*self.config['batch_size']

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score

            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.d.state_dict(), '{}{}_d_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.g.state_dict(), '{}{}_g_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.c.state_dict(), '{}{}_c_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def generate(self, datas, whole=False, samples=True, ext=False):
        self.g.eval()

        if whole:
            #prepare locations for data generation!
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            out_dirs = [out_dir + 'Z/', out_dir + 'CG_2/', out_dir + 'T/']
            if ext:
                out_dirs += [out_dir + 'Z_E/', out_dir + 'CG_2_E/', out_dir + 'T_E/']
            for o in out_dirs:
                if os.path.isdir(o):
                    shutil.rmtree(o)
                if not os.path.exists(o):
                    os.mkdir(o)

        with torch.no_grad():
            for idx, data in enumerate(datas):
                for inputs, targets, fname, _ in data:
                    # here only use 1 patch
                    inputs, targets = inputs.to(device), targets.to(device)
                    fname = os.path.basename(fname[0])
                    g_out = (self.g(inputs)).cpu().numpy().squeeze()
                    targets = targets.cpu().numpy().squeeze()

                    inputs = inputs.cpu().numpy().squeeze()
                    out = np.zeros(targets.shape)
                    out[::self.config['step_size']] = inputs

                    dir_name = self.output_dir#+str(self.epoch)+'_'
                    dir_name += fname
                    if samples:
                        self.visualize(out, g_out, targets, dir_name)
                    if whole:
                        # generate for all data
                        img_z = nib.Nifti1Image(out, np.eye(4))
                        img_z.to_filename(out_dirs[0+3*idx]+fname)

                        img_g = nib.Nifti1Image(g_out, np.eye(4))
                        img_g.to_filename(out_dirs[1+3*idx]+fname)

                        img_t = nib.Nifti1Image(targets, np.eye(4))
                        img_t.to_filename(out_dirs[2+3*idx]+fname)
                        if '0173' in fname:
                            img_z.to_filename(dir_name.replace('.nii', '_z.nii'))
                            img_g.to_filename(dir_name.replace('.nii', '_g.nii'))
                            img_t.to_filename(dir_name.replace('.nii', '_t.nii'))
                    else:
                        break

    def visualize(self, src, gen, tar, dir):
        plt.set_cmap("gray")
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig, axs = plt.subplots(3, 9, figsize=(20, 15))

        step = np.array(gen.shape)//3
        offset = step//2

        for i in range(3):

            axs[i, 0].imshow(src[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 0].set_title('Z: x_{}'.format(i), fontsize=25)
            axs[i, 0].axis('off')
            axs[i, 1].imshow(gen[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 1].set_title('G: x_{}'.format(i), fontsize=25)
            axs[i, 1].axis('off')
            axs[i, 2].imshow(tar[i*step[0]-offset[0], :, :], vmin=-1, vmax=1)
            axs[i, 2].set_title('T: x_{}'.format(i), fontsize=25)
            axs[i, 2].axis('off')

            axs[i, 3].imshow(src[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 3].set_title('Z: y_{}'.format(i), fontsize=25)
            axs[i, 3].axis('off')
            axs[i, 4].imshow(gen[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 4].set_title('G: y_{}'.format(i), fontsize=25)
            axs[i, 4].axis('off')
            axs[i, 5].imshow(tar[:, i*step[1]-offset[1], :], vmin=-1, vmax=1)
            axs[i, 5].set_title('T: y_{}'.format(i), fontsize=25)
            axs[i, 5].axis('off')

            axs[i, 6].imshow(src[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 6].set_title('Z: z_{}'.format(i), fontsize=25)
            axs[i, 6].axis('off')
            axs[i, 7].imshow(gen[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 7].set_title('G: z_{}'.format(i), fontsize=25)
            axs[i, 7].axis('off')
            axs[i, 8].imshow(tar[:, :, i*step[2]-offset[2]], vmin=-1, vmax=1)
            axs[i, 8].set_title('T: z_{}'.format(i), fontsize=25)
            axs[i, 8].axis('off')
        plt.savefig(dir.replace('nii', 'png'), dpi=150)
        plt.close()


class CNN_Wrapper:
    def __init__(self, config, model_name, exp_idx):
        self.config = config
        self.data_dir = config['data_dir']
        self.lr = config['lr']
        self.seed = exp_idx*1000
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.loss_metric = config['loss_metric']
        self.checkpoint_dir = './checkpoint_dir/'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += '{}/'.format(self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir+'output_dir/'
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        torch.manual_seed(self.seed)
        self.prepare_dataloader(config['batch_size'], self.data_dir)

        # in_size = 121*145*121
        self.cnn = _CNN(config).to(device)

        if self.loss_metric == 'Standard':
            self.criterion = nn.BCELoss().to(device)
        else:
            # self.criterion = nn.CrossEntropyLoss().to(device)
            self.criterion = nn.MSELoss().to(device)

        if config['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.cnn.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.cnn.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    def prepare_dataloader(self, batch_size, data_dir):
        train_data = B_Data(data_dir, stage='train', seed=self.seed, step_size=self.config['step_size'], Pre=self.config['pretrain'])
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        self.train_data = train_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        # self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

        valid_data = B_Data(data_dir, stage='valid', seed=self.seed, step_size=self.config['step_size'], Pre=self.config['pretrain'])
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

        test_data  = B_Data(data_dir, stage='test', seed=self.seed, step_size=self.config['step_size'], Pre=self.config['pretrain'])
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = B_Data(data_dir, stage='all', seed=self.seed, step_size=self.config['step_size'], Pre=self.config['pretrain'])
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))

        Data_dir_NACC = data_dir[:-1] + '_E/'
        external_data = B_Data(Data_dir_NACC, stage='all', seed=self.seed, step_size=self.config['step_size'], external=True)
        self.external_data = external_data
        self.ext_dataloader = DataLoader(self.external_data, batch_size=1)

    def load(self, dir, fixed=False):
        print('loading pre-trained model...')
        dir = glob.glob(dir + '*.pth')
        st = torch.load(dir[0])
        # del st['l2.weight']
        # del st['l2.bias']
        self.cnn.load_state_dict(st, strict=False)
        if fixed:
            # need update if want to use
            ps = []
            for n, p in self.model.named_parameters():
                if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # for n, p in self.model.named_parameters():
            # print(n, p.requires_grad)
        print('loaded.')

    def train(self, epochs, training_prints=3):
        print('training ... (seed={})'.format(self.seed))
        torch.use_deterministic_algorithms(True)
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        self.epoch = 0

        valid_losses, train_losses = [], []
        for self.epoch in range(1, epochs+1):
            train_loss = self.train_model_epoch()
            rec = 0
            if self.epoch % 10 == 0:
                rec = 1
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//training_prints) == 0:
                rec = 1
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)

                end.record()
                torch.cuda.synchronize()
                print('{}th epoch validation loss [{}] ='.format(self.epoch, self.config['loss_metric']), '%.3f' % (val_loss), '|| train loss =', '%.3f' % (train_loss), '|| time(s) =', start.elapsed_time(end)//1000)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            if rec:
                valid_losses += [val_loss]
                train_losses += [train_loss]

        print('Best model valid loss:', self.optimal_valid_metric.item(), self.optimal_epoch)
        # print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())
        # print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        self.plot_train(valid_losses, train_losses)
        return self.optimal_valid_metric

    def plot_train(self, d_rs, d_gs):
        plt.figure(figsize=(10,5))
        plt.title('Training Loss')
        plt.plot(d_rs, label='valid')
        plt.plot(d_gs, label='train')
        plt.xlabel("epochs")
        plt.ylabel("preds")
        plt.legend()
        # plt.show()
        plt.savefig(self.output_dir+'train_loss.png', dpi=150)
        plt.close()

    def train_model_epoch(self):
        self.cnn.train(True)
        train_loss = []

        # torch.use_deterministic_algorithms(False)
        for _, inputs, _, labels in self.train_dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            self.cnn.zero_grad()

            preds = self.cnn(inputs).view(-1)
            loss = self.criterion(preds, labels)
            loss.backward()
            train_loss += [preds.mean().item()]
            self.optimizer.step()
            # clip = 1
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        # torch.use_deterministic_algorithms(True)
        return np.mean(train_loss)/self.config['batch_size']

    def valid_model_epoch(self):
        self.cnn.eval()
        with torch.no_grad():
            loss_all = []
            for _, inputs, _, labels in self.valid_dataloader:
                # here only use 1 patch
                inputs, labels = inputs.to(device), labels.float().to(device)
                preds = self.cnn(inputs).view(-1)
                loss = self.criterion(preds, labels)
                loss_all += [loss.item()]
        return np.mean(loss_all)

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score

            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.cnn.state_dict(), '{}{}_cnn_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def test(self, out=False, key='test'):
        # if out is True, return the report dictionary; else print report
        # only look at one item with specified key; i.e. test dataset
        self.cnn.eval()
        dls = [self.train_dataloader, self.valid_dataloader, self.test_dataloader, self.ext_dataloader]
        names = ['train dataset', 'valid dataset', 'test dataset', 'ext dataset']
        target_names = ['class ' + str(i) for i in range(2)]
        for dl, n in zip(dls, names):
            if key not in n:
                continue

            preds_raw = []
            preds_all = []
            labels_all = []
            with torch.no_grad():
                for _, inputs, _, labels in dl:
                    # here only use 1 patch
                    inputs, labels = inputs.to(device), labels.float().to(device)
                    preds_raw += self.cnn(inputs).view(-1).cpu()
                    preds_all += torch.round(self.cnn(inputs).view(-1)).cpu()
                    labels_all += labels.cpu()
            f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(key, self.exp_idx), 'w')
            write_raw_score(f, preds_raw, labels_all)
            f.close()

            if out:
                report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1], target_names=target_names, zero_division=0, output_dict=True)
                return report
            else:
                report = classification_report(y_true=labels_all, y_pred=preds_all, labels=[0,1], target_names=target_names, zero_division=0, output_dict=False)
                print(n)
                print(report)


class AE_Wrapper:
    def __init__(self, fil_num, drop_rate, seed, batch_size, balanced, Data_dir, exp_idx, num_fold, model_name, metric, patch_size, lr, augment=False, dim=1, yr=2, loss_v=0):
        self.loss_imp = 0.0
        self.loss_tot = 0.0
        self.seed = seed
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.Data_dir = Data_dir
        self.patch_size = patch_size
        self.model_name = model_name
        self.augment = augment
        self.cox_local = 1
        #'macro avg' or 'weighted avg'
        self.eval_metric = metric
        self.dim = dim
        torch.manual_seed(seed)

        fil_num = 30 #either this or batch size
        # in_size = 167*191*167
        vector_len = 3
        # fil_num = 512
        # self.model = _FCN(num=fil_num, p=drop_rate, dim=self.dim, out=1).cuda()
        self.encoder = _Encoder(drop_rate=.5, fil_num=fil_num, out_channels=vector_len).cuda()
        self.decoder = _Decoder(drop_rate=.5, fil_num=fil_num, in_channels=vector_len).cuda()
        # self.decoder = _Decoder(in_size=out, drop_rate=.5, out=in_size, fil_num=fil_num).cuda()

        self.yr=yr
        self.prepare_dataloader(batch_size, balanced, Data_dir)
        # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
        # self.criterion = cox_loss
        self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizerE = optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizerD = optim.Adam(self.decoder.parameters(), lr=lr)
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.DPMs_dir = './DPMs/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.DPMs_dir):
            os.mkdir(self.DPMs_dir)
        if os.path.isdir('ae_valid/'):
            shutil.rmtree('ae_valid/')
        os.mkdir('ae_valid/')

    def prepare_dataloader(self, batch_size, balanced, Data_dir):
        if self.augment:
            train_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=Augment(), dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        else:
            train_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='train', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        if balanced == 1:
            sample_weight = [sample_weight[i] for i in train_data.index_list]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
            self.imbalanced_ratio = 1
        elif balanced == 0:
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = AE_Cox_Data(Data_dir, self.exp_idx, stage='valid_patch', seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)

        all_data = AE_Cox_Data(Data_dir, self.exp_idx, stage='all', seed=self.seed, patch_size=self.patch_size, transform=None, dim=self.dim, name=self.model_name, fold=self.num_fold, yr=self.yr)
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))
        self.train_all_dataloader = DataLoader(all_data, batch_size=len(train_data))

    def get_info(self, stage, debug=False):
        all_x, all_obss, all_hits = [],[],[]
        if stage == 'train':
            dl = self.train_all_dataloader
        elif stage == 'all':
            dl = self.all_dataloader
        else:
            raise Exception('Error in fn get info: stage unavailable')
        for items in dl:
            all_x, all_obss, all_hits = items
        idxs = torch.argsort(all_obss, dim=0, descending=True)
        all_x = all_x[idxs]
        all_obss = all_obss[idxs]
        # all_hits = all_hits[idxs]
        with torch.no_grad():
            h_x = self.model(all_x.cuda())

        all_logs = torch.log(torch.cumsum(torch.exp(h_x), dim=0))
        if debug:
            print('h_x', h_x[:10])
            print('exp(h_x)', torch.exp(h_x))
            print('cumsum(torch.exp(h_x)', torch.cumsum(torch.exp(h_x), dim=0))
            print('all_logs', all_logs)
        return all_logs, all_obss

    def train_model_epoch(self):
        self.encoder.train(True)
        self.decoder.train(True)
        for inputs, obss, hits in self.train_dataloader:
            inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()

            # idxs = torch.argsort(obss, dim=0, descending=True)
            # inputs = inputs[idxs]
            # obss = obss[idxs]
            # hits = hits[idxs]

            # if torch.sum(hits) == 0:
            #     continue # because 0 indicates nothing to learn in this batch, we skip it
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            # inputs = inputs.view(inputs.shape[0], -1)

            vector = self.encoder(inputs)
            outputs = self.decoder(vector)
            loss = self.criterion(outputs, inputs)
            loss.backward()

            self.optimizerE.step()
            self.optimizerD.step()

            # vector = self.encoder(inputs)
            # outputs = self.decoder(vector)
            # loss2 = self.criterion(outputs, inputs)
            # if loss2 < loss:
                # self.loss_imp += 1
            # self.loss_tot += 1

    def train(self, epochs):
        self.optimal_valid_matrix = None
        # self.optimal_valid_matrix = np.array([[0, 0, 0, 0]]*4)
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        for self.epoch in range(epochs):
            self.train_model_epoch()
            if self.epoch % 20 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                # print('{}th epoch validation loss:'.format(self.epoch), '[MSE-based]:', '%.4f, loss improved: %.2f' % (val_loss, self.loss_imp/self.loss_tot))
                print('{}th epoch validation loss [MSE-based]:'.format(self.epoch), '%.4f' % (val_loss))
                # if self.epoch % (epochs//10) == 0:
                #     print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix.tolist(), 'eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric)
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def save_checkpoint(self, loss):
        # if self.eval_metric(valid_matrix) >= self.optimal_valid_metric:
        # need to modify the metric
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            # torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.encoder.state_dict(), '{}{}_{}_en.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
            torch.save(self.decoder.state_dict(), '{}{}_{}_de.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))

    def valid_model_epoch(self):
        with torch.no_grad():
            self.encoder.train(False)
            self.decoder.train(False)
            # loss_all = 0
            patches_all = []
            obss_all = []
            hits_all = []
            # for patches, labels in self.valid_dataloader:
            for data, obss, hits in self.valid_dataloader:
                # if torch.sum(hits) == 0:
                    # continue # because 0 indicates nothing to learn in this batch, we skip it
                patches, obs, hit = data, obss, hits

                # here only use 1 patch
                # patch = patches[0].reshape(-1)
                patch = patches
                patches_all += [patch]
                # patches_all += [patch.numpy()]
                # obss_all += [obss.numpy()[0]]
                # hits_all += [hits.numpy()[0]]
                obss_all += [obss]
                hits_all += [hits]

            # idxs = np.argsort(obss_all, axis=0)[::-1]
            patches_all = np.array(patches_all)
            obss_all = np.array(obss_all)
            hits_all = np.array(hits_all)

            patches_all = torch.tensor(patches_all)

            preds_all = self.decoder(self.encoder((patches_all.cuda()))).cpu()
            # preds_all, obss_all, hits_all = preds_all.view(-1, 1).cuda(), torch.tensor(obss_all).view(-1).cuda(), torch.tensor(hits_all).view(-1).cuda()

            loss = self.criterion(preds_all, patches_all)

        with torch.no_grad():
            number = 2
            plt.figure(figsize=(20, 4))
            for index in range(number):
                # display original
                ax = plt.subplot(2, number, index + 1)
                plt.imshow(patches_all[index].cpu().numpy()[0,60])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, number, index + 1 + number)
                plt.imshow(preds_all[index].cpu().numpy()[0,60])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()
            plt.savefig('ae_valid/'+str(self.epoch)+"AE.png")
            plt.close()
        return loss

    def test_and_generate_DPMs(self, epoch=None, stages=['train', 'valid', 'test'], single_dim=True, root=None, upsample=True, CSV=True):
        if epoch:
            self.encoder.load_state_dict(torch.load('{}{}_{}_en.pth'.format(self.checkpoint_dir, self.model_name, epoch)))
            self.decoder.load_state_dict(torch.load('{}{}_{}_de.pth'.format(self.checkpoint_dir, self.model_name, epoch)))
        else:
            dir1 = glob.glob(self.checkpoint_dir + '*_en.pth')
            dir2 = glob.glob(self.checkpoint_dir + '*_de.pth')
            self.encoder.load_state_dict(torch.load(dir1[0]))
            self.decoder.load_state_dict(torch.load(dir2[0]))
        print('testing and generating DPMs ... ')
        if root:
            print('\tcustom root directory detected:', root)
            self.DPMs_dir = self.DPMs_dir.replace('./', root)
        if os.path.isdir(self.DPMs_dir):
            shutil.rmtree(self.DPMs_dir)
        os.mkdir(self.DPMs_dir)
        # self.fcn = self.model.dense_to_conv()
        self.fcn = self.encoder
        self.fcn.train(False)
        with torch.no_grad():
            if single_dim:
                if os.path.isdir(self.DPMs_dir+'1d/'):
                    shutil.rmtree(self.DPMs_dir+'1d/')
                os.mkdir(self.DPMs_dir+'1d/')
                if os.path.isdir(self.DPMs_dir+'upsample_vis/'):
                    shutil.rmtree(self.DPMs_dir+'upsample_vis/')
                os.mkdir(self.DPMs_dir+'upsample_vis/')
                if os.path.isdir(self.DPMs_dir+'upsample/'):
                    shutil.rmtree(self.DPMs_dir+'upsample/')
                os.mkdir(self.DPMs_dir+'upsample/')
                if os.path.isdir(self.DPMs_dir+'nii_format/'):
                    shutil.rmtree(self.DPMs_dir+'nii_format/')
                os.mkdir(self.DPMs_dir+'nii_format/')
            for stage in stages:
                Data_dir = self.Data_dir
                if stage in ['AIBL', 'NACC']:
                    Data_dir = Data_dir.replace('ADNI', stage)
                data = AE_Cox_Data(Data_dir, self.exp_idx, stage=stage, whole_volume=True, seed=self.seed, patch_size=self.patch_size, name=self.model_name, fold=self.num_fold, yr=self.yr)
                fids = data.index_list
                filenames = data.Data_list
                dataloader = DataLoader(data, batch_size=1, shuffle=False)
                DPMs, Labels = [], []
                labels_all = []
                print('len(data)', len(data), stage)
                for idx, (inputs, obss, hits) in enumerate(dataloader):
                    labels_all += hits.tolist()
                    inputs, obss, hits = inputs.cuda(), obss.cuda(), hits.cuda()

                    # inputs = inputs.view(inputs.shape[0], -1)
                    DPM_tensor = self.fcn(inputs)
                    DPM = DPM_tensor.cpu().numpy().squeeze()
                    if single_dim:
                        m = nn.Softmax(dim=1) # dim=1, as the output shape is [1, 2, cube]
                        n = nn.LeakyReLU()
                        DPM2 = m(DPM_tensor).cpu().numpy().squeeze()
                        # DPM2 = m(DPM_tensor).cpu().numpy().squeeze()[1]
                        # print(np.argmax(DPM, axis=0))
                        v = nib.Nifti1Image(DPM2, np.eye(4))
                        nib.save(v, self.DPMs_dir + 'nii_format/' + os.path.basename(filenames[fids[idx]]))

                        DPM3 = n(DPM_tensor).cpu().numpy().squeeze()
                        # DPM3 = DPM_tensor.cpu().numpy().squeeze() #might produce strange edges on the side, see later comments

                        DPM3 = np.around(DPM3, decimals=2)
                        np.save(self.DPMs_dir + '1d/' + os.path.basename(filenames[fids[idx]]), DPM2)
                        if upsample:
                            DPM_ni = nib.Nifti1Image(DPM3, np.eye(4))
                            # shape = list(inputs.shape[2:])
                            # [167, 191, 167]
                            shape = [121, 145, 121] # fixed value here, because the input is padded, thus cannot be used here

                            # if not using the activation fn, need to subtract a small value to offset the boarder of the resized image
                            # vals = np.append(np.array(DPM_ni.shape)/np.array(shape)-0.005,[1])
                            vals = np.append(np.array(DPM_ni.shape)/np.array(shape),[1])

                            affine = np.diag(vals)
                            DPM_ni = nilearn.image.resample_img(img=DPM_ni, target_affine=affine, target_shape=shape)
                            nib.save(DPM_ni, self.DPMs_dir + 'upsample/' + os.path.basename(filenames[fids[idx]]))
                            DPM_ni = DPM_ni.get_data()

                            plt.set_cmap("jet")
                            plt.subplots_adjust(wspace=0.3, hspace=0.3)
                            fig, axs = plt.subplots(3, 3, figsize=(20, 15))
                            # fig, axs = plt.subplots(3, 3, figsize=(20, 15))
                            # fig, axs = plt.subplots(2, 3, figsize=(40, 30))

                            INPUT = inputs.cpu().numpy().squeeze()

                            slice1, slice2, slice3 = DPM_ni.shape[0]//2, DPM_ni.shape[1]//2, DPM_ni.shape[2]//2
                            slice1b, slice2b, slice3b = INPUT.shape[0]//2, INPUT.shape[1]//2, INPUT.shape[2]//2
                            slice1c, slice2c, slice3c = DPM3.shape[0]//2, DPM3.shape[1]//2, DPM3.shape[2]//2

                            axs[0,0].imshow(DPM_ni[slice1, :, :].T)
                            # print(DPM_ni[slice1, :, :].T)
                            # axs[0,0].imshow(DPM_ni[slice1, :, :].T, vmin=0, vmax=1)
                            axs[0,0].set_title('output. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[0,0].axis('off')
                            im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T)
                            # im1 = axs[0,1].imshow(DPM_ni[:, slice2, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[0,1].axis('off')
                            im = axs[0,2].imshow(DPM_ni[:, :, slice3].T)
                            # axs[0,2].imshow(DPM_ni[:, :, slice3].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[0,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[0,2])
                            cbar.ax.tick_params(labelsize=20)

                            axs[1,0].imshow(INPUT[slice1b, :, :].T)
                            # axs[1,0].imshow(DPM3[slice1b, :, :].T, vmin=0, vmax=1)
                            axs[1,0].set_title('input. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[1,0].axis('off')
                            axs[1,1].imshow(INPUT[:, slice2b, :].T)
                            # axs[1,1].imshow(DPM3[:, slice2b, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[1,1].axis('off')
                            im = axs[1,2].imshow(INPUT[:, :, slice3b].T)
                            # axs[1,2].imshow(DPM3[:, :, slice3b].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[1,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[1,2])
                            cbar.ax.tick_params(labelsize=20)

                            axs[2,0].imshow(DPM3[slice1c, :, :].T)
                            # axs[2,0].imshow(DPM3[slice1c, :, :].T, vmin=0, vmax=1)
                            axs[2,0].set_title('output small size. status:'+str(hits.cpu().numpy().squeeze()), fontsize=25)
                            axs[2,0].axis('off')
                            axs[2,1].imshow(DPM3[:, slice2c, :].T)
                            # axs[2,1].imshow(DPM3[:, slice2c, :].T, vmin=0, vmax=1)
                            # axs[1].set_title('v2', fontsize=25)
                            axs[2,1].axis('off')
                            im = axs[2,2].imshow(DPM3[:, :, slice3c].T)
                            # axs[2,2].imshow(DPM3[:, :, slice3c].T, vmin=0, vmax=1)
                            # axs[2].set_title('v3', fontsize=25)
                            axs[2,2].axis('off')
                            cbar = fig.colorbar(im, ax=axs[2,2])
                            cbar.ax.tick_params(labelsize=20)

                            # cbar = fig.colorbar(im1, ax=axs)
                            # cbar.ax.tick_params(labelsize=20)
                            plt.savefig(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png', dpi=150)
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            plt.close()
                            # print(self.DPMs_dir + 'upsample_vis/' + os.path.basename(filenames[fids[idx]]) + '.png')
                            # if idx == 10:
                            #     sys.exit()
                    np.save(self.DPMs_dir + os.path.basename(filenames[fids[idx]]), DPM)
                    DPMs.append(DPM)
                    Labels.append(hits)

                if CSV:
                    rids = list(data.fileIDs)
                    filename = '{}_{}_{}'.format(self.exp_idx, stage, self.model_name) #exp_stage_model-scan
                    with open('fcn_csvs/'+filename+'.csv', 'w') as f:
                        wr = csv.writer(f)
                        wr.writerows([['label']+labels_all]+[['RID']+rids])
                # matrix, ACCU, F1, MCC = DPM_statistics(DPMs, Labels)
                # np.save(self.DPMs_dir + '{}_MCC.npy'.format(stage), MCC)
                # np.save(self.DPMs_dir + '{}_F1.npy'.format(stage),  F1)
                # np.save(self.DPMs_dir + '{}_ACCU.npy'.format(stage), ACCU)
                # print(stage + ' confusion matrix ', matrix, ' accuracy ', self.eval_metric(matrix))
        # print(DPM.shape)

        print('DPM generation is done')

    def predict():
        # TODO: given testing data, produce survival plot, based on time
        # could be a single patient
        return

class MLP_Wrapper:
    def __init__(self, 
                 exp_idx,
                 model_name,
                 lr,
                 weight_decay,
                 model,
                 model_kwargs,
                 add_age=False,
                 add_mmse=False):
        self._age = add_age
        self._mmse = add_mmse
        self.seed = exp_idx
        self.exp_idx = exp_idx
        self.model_name = model_name
        self.device = device
        self.dataset = ParcellationDataBinary
        self.lr = lr
        self.weight_decay = weight_decay
        self.c_index = []
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, self.exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.prepare_dataloader()
        torch.manual_seed(self.seed)
        self.criterion = nn.BCELoss().to(self.device)
        self.model = model(in_size=self.in_size, **model_kwargs).float()
        self.model.to(self.device)

    def prepare_dataloader(self):
        kwargs = dict(exp_idx=self.exp_idx,
                    add_age=self._age,
                    add_mmse=self._mmse)
        self.train_data = self.dataset(stage = 'train', dataset='ADNI', **kwargs)
        self.features = self.train_data.get_features()
        self.in_size = len(self.features)
        self.valid_data = self.dataset(stage = 'valid', dataset='ADNI', **kwargs)
        self.test_data = self.dataset(stage = 'test', dataset='ADNI', **kwargs)
        self.all_data = self.dataset(stage= 'all', dataset='ADNI', **kwargs)
        self.nacc_data = self.dataset(stage= 'all', dataset='NACC', **kwargs)
        self.train_dataloader = DataLoader(self.train_data, batch_size=len(self.train_data))
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=len(self.valid_data))
        self.test_dataloader = DataLoader(self.test_data, batch_size=len(self.test_data))
        self.all_dataloader = DataLoader(self.all_data, batch_size=len(self.all_data),
                                         shuffle=False)
        self.nacc_dataloader = DataLoader(self.nacc_data, batch_size=len(self.nacc_data))

    def load(self):
        file = glob.glob(self.checkpoint_dir + self.model_name + '*')
        print(file)
        assert(len(file)) == 1
        self.model.load_state_dict(torch.load(file[0]))

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, _, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(),
                       '{}{}_{}.pth'.format(
                    self.checkpoint_dir, self.model_name, self.optimal_epoch)
                       )

    def train(self, epochs):
        self.val_loss = []
        self.train_loss = []
        self.optimal_valid_matrix = None
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(
                0.5, 0.999), weight_decay=self.weight_decay)
        for self.epoch in range(epochs):
                self.train_model_epoch(self.optimizer)
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
                self.val_loss.append(val_loss)
                if self.epoch % 300 == 0:
                    print('{}: {}th epoch validation score: {}'.format(
                            self.model_name, self.epoch, val_loss))
        print('Best model saved at the {}th epoch; cox-based loss: {}'.format(
                self.optimal_epoch, self.optimal_valid_metric))
        self.optimal_path = '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch)
        print('Location: '.format(self.optimal_path))
        return self.optimal_valid_metric

    def train_model_epoch(self, optimizer):
        self.model.train()
        for data, pmci, _ in self.train_dataloader:
            self.model.zero_grad()
            preds = self.model(data.to(self.device).float())
            loss = self.criterion(preds.squeeze(),pmci.to(self.device).float().squeeze())
            loss.backward()
            optimizer.step()

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.eval()
            for data, pmci, _ in self.valid_dataloader:
                preds = self.model(data.to(self.device).float())
                loss = self.criterion(preds.squeeze(),pmci.to(self.device).float().squeeze())
        return loss
        
    def retrieve_testing_data(self, external_data):
        if external_data:
            dataloader = self.nacc_dataloader
        else:
            dataloader = self.test_dataloader
        with torch.no_grad():
            self.load()
            self.model.eval()
            for data, pmci, rids in dataloader:
                preds = self.model(data.to(self.device).float()).to('cpu')
                preds = torch.round(preds)
                rids = rids
            return preds, pmci, rids

    def test_surv_data_optimal_epoch(self, external_data=False):
        key = ['ADNI' if not external_data else 'NACC']
        preds, pmci, rids = self.retrieve_testing_data(external_data)                             
        f = open(self.checkpoint_dir + 'raw_score_{}_{}.txt'.format(key, self.exp_idx), 'w')
        write_raw_score(f, preds, pmci)
        f.close()
        report = classification_report(
            y_true=pmci,
            y_pred=preds, labels=[0,1],
            target_names= key,
            zero_division=0, output_dict=False)
        print(report)


def _test():
    mlp_list = []
    for exp in range(5):
        mlp = MLP_Wrapper(
            exp_idx=exp,
            model_name = f'mlp_bce_{exp}',
            lr = 0.01,
            weight_decay=0,
            model=_MLP_Surv,
            model_kwargs=dict(
                drop_rate=0.5,
                fil_num=100,
                output_shape=1)
        )
        mlp_list.append(mlp)
    for mlp in mlp_list:
        mlp.train(1000)
        mlp.test_surv_data_optimal_epoch(external_data=True)

if __name__ == "__main__":
    _test()