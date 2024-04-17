from email import message
import shutil
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
from Dataloader import ITMVQA_train
from torch.utils.data import DataLoader
from torch.nn import init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import *
import os
import random
from scipy import stats

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class ITMVQA_demo:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids  # 所有gpu号
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        self.lr = opt.lr
        self.gamma = opt.gamma
        self.milestones = opt.milestones
        self.save_dir = opt.save_dir + 'VQA_model/'
        self.load_dir = opt.load_dir
        self.mode = opt.mode
        self.G_n = opt.G_n
        self.D_n = opt.D_n
        self.train_status()
        self.tb = SummaryWriter(opt.logdir+'ITMVQA')
        self.tb_step = 0
        self.tbnum = opt.tbnum
        

    def train_status(self):
        if self.mode == 'GDGTDT':
            self.model = IQA_net_GDGTDT(self.G_n, self.D_n).to(self.device)  # 模型迁移到第一个GPU
        elif self.mode == 'GGT':
            self.model = IQA_net_GGT().to(self.device)  # 模型迁移到第一个GPU
        elif self.mode == 'DDT':
            self.model = IQA_net_GC().to(self.device)  # 模型迁移到第一个GPU
        elif self.mode == 'GD':
            self.model = IQA_net_GDC().to(self.device)  # 模型迁移到第一个GPU
        elif self.mode == 'GTDT':
            self.model = IQA_net_GDC_T().to(self.device)  # 模型迁移到第一个GPU
        else: assert self.mode == False
        self.set_loss_optimizer_scheduler()
        self.load_network()
        self.model = self.model.to(self.device)
        self.model.train()

    def set_loss_optimizer_scheduler(self):
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.sche = lr_scheduler.MultiStepLR(self.optim, milestones=self.milestones, gamma=self.gamma)
        self.L1loss = nn.L1Loss().to(self.device)

    def init_weight(self, net, init_type):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data)
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data)
                else:
                    raise NotImplementedError('initialization method {} is not implemented'.format(init_type))
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data)
                init.constant_(m.bias.data, 0.0)

        print('--initialize network with {}'.format(init_type))
        net.apply(init_func)

    def load_network(self):
        self.init_weight(self.model, 'xavier')
        if self.load_dir is not None:
            checkpoint = torch.load(self.load_dir, map_location=self.device)
            self.model.load_state_dict(checkpoint['ITM_Net'])
            print('--完成权重加载:{}--'.format(self.load_dir))

    # 训练过程中相关操作
    def train_step(self, data):
        self.model.train()
        # set_train_data
        # self.feat = data['feat'].to(self.device)
        self.mos_sdr = data['mos_sdr'].to(self.device)
        self.G_feat_A = data['G_feat_A'].to(self.device)
        self.D_feat_A = data['D_feat_A'].to(self.device)
        self.G_feat_V = data['G_feat_V'].to(self.device)
        self.D_feat_V = data['D_feat_V'].to(self.device)
        # self.feat = {'G_feat':self.G_feat,'C_feat':self.C_feat}
        self.feat = {'G_feat_A': self.G_feat_A,
                     'D_feat_A': self.D_feat_A,
                     'G_feat_V': self.G_feat_V,
                     'D_feat_V': self.D_feat_V,
                     'mos_sdr': self.mos_sdr}
        self.MOS = data['mos'].to(self.device)

        # cal_loss
        # self.pred = self.model(self.feat)
        self.pred = self.model(self.feat)
        self.loss = self.L1loss(self.pred, self.MOS)

        self.optim.zero_grad()  # 梯度置零
        self.loss.backward()
        self.optim.step()  # 更新参数
        
    def test_step(self, data):
        self.model.eval()
        # set_train_data
        with torch.no_grad():
            self.mos_sdr = data['mos_sdr'].to(self.device)
            self.G_feat_A = data['G_feat_A'].to(self.device)
            self.D_feat_A = data['D_feat_A'].to(self.device)
            self.G_feat_V = data['G_feat_V'].to(self.device)
            self.D_feat_V = data['D_feat_V'].to(self.device)
            # self.feat = {'G_feat':self.G_feat,'C_feat':self.C_feat}
            self.feat = {'G_feat_A': self.G_feat_A,
                        'D_feat_A': self.D_feat_A,
                        'G_feat_V': self.G_feat_V,
                        'D_feat_V': self.D_feat_V,
                        'mos_sdr': self.mos_sdr}
            self.MOS = data['mos'].to(self.device)
            self.pred = self.model(self.feat)

    def get_current_lr(self):
        optimizers = [self.optim]
        return [param_group['lr'] for param_group in optimizers[0].param_groups]

    def schedulers_step(self):
        self.sche.step()

    def save_network(self, epoch):
        mkdir(self.save_dir)
        save_path = self.save_dir + 'ITMVQA_model_{}.pth'.format(self.vname)
        state = {'ITMVQA_model': self.model.state_dict(),
                 'optim': self.optim.state_dict(),
                 'epoch': epoch}
        torch.save(state, save_path)

    def tensorboard(self):
        self.tb_step += 1
        loss = self.loss.item()
        self.tb.add_scalars('loss', {'loss': loss}, self.tb_step)
        return loss

def readlink(opt):
    # F = open(opt.linkpath, 'r')
    # train_link = []
    # test_link = []
    # text = F.read().split('\n')
    # for t in range(len(text)//2):
    #     train_link = train_link + [[int(f) for f in text[2*t].split(',')]]
    #     test_link = test_link + [[int(f) for f in text[2*t+1].split(',')]]
    train_link, test_link = [], []
    random.seed(2022)
    for i in range(10):
        link = np.arange(0,20).tolist()
        random.shuffle(link)
        train_link.append(link[:16])
        test_link.append(link[16:])
    # print(train_link)
    
    return train_link, test_link
    
def ITMVQA_FC(opt):
    torch.manual_seed(2022)
    random.seed(2022)
    best_scc_list = []
    best_pcc_list = []
    best_rmse_list = []
    best_kcc_list = []
    train_best_scc_list = []
    train_best_pcc_list = []
    train_link_, test_link_ = readlink(opt)
    for test_num in range(opt.train_test_num):
        # link = list(range(20))
        # random.shuffle(link)
        # train_link = link[:int(len(link)*0.8)]
        # test_link = link[int(len(link)*0.8):]

        train_link = train_link_[test_num]
        test_link = test_link_[test_num]
        train_loader = DataLoader(ITMVQA_train(train_link, opt), num_workers=opt.num_workers, batch_size = opt.batch_size, shuffle=True)
        test_loader = DataLoader(ITMVQA_train(test_link, opt), batch_size = 1)
        model = ITMVQA_demo(opt)
        scc_max = 0
        pcc_max = 0
        for epoch in range(opt.epoch_start, opt.epoch_end + 1):
            losses = []
            lr = model.get_current_lr()
            pred_scores = []
            gt_scores = []
            test_pred = []
            test_mos = []
            for i, data in enumerate(train_loader):
                
                model.train_step(data)
                loss = model.tensorboard()
                losses.append(loss)
                pred_scores = pred_scores + model.pred.cpu().tolist()
                gt_scores = gt_scores + model.MOS.cpu().tolist()
            # print(data['G_feat_A'].shape)    
            for i, data in enumerate(test_loader):
                model.test_step(data)
                test_pred = test_pred + model.pred.cpu().tolist()
                test_mos = test_mos + model.MOS.cpu().tolist()

            pred_scores = np.reshape(pred_scores, -1)
            gt_scores = np.reshape(gt_scores, -1)
            test_pred = np.reshape(test_pred, -1)
            test_mos = np.reshape(test_mos, -1)
            # print(len(pred_scores), len(gt_scores))
            # print(pred_scores, gt_scores)
            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            # print(len(pred_scores), len(gt_scores))
            # print(len(test_pred), len(test_mos))
            train_pcc, _ = stats.pearsonr(pred_scores, gt_scores)
            #平均为1个分数对应一张图
            #        
            test_srcc,_ = stats.spearmanr(test_pred, test_mos)
            test_pcc,_ = stats.pearsonr(test_pred, test_mos)
            test_RMSE = np.sqrt(((test_pred - test_mos)**2).mean())
            test_KROCC,_ = stats.kendalltau(test_pred, test_mos)
            
            epoch_message = 'train_test_num:{}, lr:{}, losses:{}, epoch:{}, train_srcc:{}, train_pcc:{}, test_srcc:{}, test_pcc:{}'.format(test_num+1, lr, sum(losses)/len(losses),
                                                                                                                         epoch, train_srcc, train_pcc,
                                                                                                                         test_srcc, test_pcc)
            print(epoch_message)
            F = open(opt.loss_file, 'a', encoding='utf-8')
            F.write(epoch_message)
            F.write('\n') 
            F.close()

            if test_srcc+test_pcc > scc_max+pcc_max:
                scc_max = test_srcc
                pcc_max = test_pcc
                rmse_max = test_RMSE
                kcc_max = test_KROCC
                train_srcc_ = train_srcc
                train_pcc_ = train_pcc
                test_mos_ = test_mos
                test_pred_ =  test_pred
                lr_ = lr
                epoch_ = epoch
                epoch_message_best = 'train_test_num:{}, lr:{}, epoch:{}, best_scc:{}, best_pcc:{}, best_rmse:{}, best_kcc:{}, train_srcc:{}, train_pcc:{}'.format(test_num+1, lr,
                                                                                                                         epoch, scc_max, pcc_max, rmse_max, kcc_max,
                                                                                                                         train_srcc, train_pcc
                                                                                                                         )
                print(epoch_message_best, '\n')
                
            model.schedulers_step()
        F_pred = open(opt.record_pred, 'a', encoding='utf-8')
        F_pred.write(epoch_message_best)
        F_pred.write('\n')
        F_pred.write('test_mos')
        for i in range(len(test_mos_)):
            if i%10 == 0:
                F_pred.write(str(i//10 + 1))
                F_pred.write('\n')
            F_pred.write(str(test_mos_[i])+' ')
            F_pred.write(str(test_pred_[i])+' ')
            F_pred.write('\n')
        F_pred.write('\n')
        F_pred.close()
                
        best_scc_list.append(scc_max)
        best_pcc_list.append(pcc_max)
        best_rmse_list.append(rmse_max)
        best_kcc_list.append(kcc_max)

        train_best_scc_list.append(train_srcc_)
        train_best_pcc_list.append(train_pcc_)
        print('best_scc_mean:{}, best_pcc_mean:{}'.format(np.mean(best_scc_list), np.mean(best_pcc_list)))

        epoch_message_best = 'train_test_num:{}, lr:{}, epoch:{}, best_scc:{}, best_pcc:{}, train_srcc:{}, train_pcc:{}'.format(test_num+1, lr_,
                                                                                                                         epoch_, scc_max, pcc_max, 
                                                                                                                         train_srcc_, train_pcc_)
        linkmessage = 'train_link:{}, test_link:{}'.format(train_link, test_link)
        F = open(opt.loss_file_best, 'a', encoding='utf-8')
        F.write(linkmessage)
        F.write('\n') 
        F.write(epoch_message_best)
        F.write('\n') 
        if test_num == opt.train_test_num-1:
            train_test_message = 'best_scc_mean:{}, best_pcc_mean:{}, best_rmse_mean:{}, best_kcc_mean:{}, train_best_scc_mean:{}, train_best_pcc_mean:{}'.format(np.mean(best_scc_list), 
                                                                                                                        np.mean(best_pcc_list),
                                                                                                                        np.mean(best_rmse_list),
                                                                                                                        np.mean(best_kcc_list),
                                                                                                                        np.mean(train_best_scc_list),
                                                                                                                        np.mean(train_best_pcc_list))
            F.write(train_test_message)
        F.close()
        # model.save_network(epoch=epoch)
        model.tb.close()

# random.seed(901)
# link = list(range(20))
# random.shuffle(link)
# test_link = ['V'+str(i) for i in link[:int(len(link)*0.8)]]
# train_link = ['V'+str(i) for i in link[int(len(link)*0.8):]]
# print(test_link)
# print(train_link)
            
