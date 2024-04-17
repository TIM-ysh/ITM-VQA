import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import kornia
import numpy as np
from Dataloader import finetune_Dataset
from torch.utils.data import DataLoader
from torch.nn import init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import ITM_net, G_model
import os

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class Net_featureExact_G:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids  # 所有gpu号
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        self.lr = opt.lr
        self.gamma = opt.gamma
        self.milestones = opt.milestones
        self.save_dir = opt.save_dir + 'G_model/'
        self.G_txt = opt.G_txt
        self.load_dir = opt.load_dir_G
        self.base_nf = opt.G_base_nf 
        self.train_status()
        self.tb = SummaryWriter(opt.logdir+'G')
        self.tb_step = 0
        self.tbnum = opt.tbnum
        self.vname = opt.train_hdr.split('/')[-1]
        

    def train_status(self):
        self.model = G_model(self.base_nf).to(self.device)  # 模型迁移到第一个GPU
        self.set_loss_optimizer_scheduler()
        self.load_network()
        self.model = self.model.to(self.device)
        self.model.train()

    def set_loss_optimizer_scheduler(self):
        self.optim_hdr = optim.Adam(self.model.parameters(), lr=self.lr)
        self.sche_hdr = lr_scheduler.MultiStepLR(self.optim_hdr, milestones=self.milestones, gamma=self.gamma)
        self.MSE = nn.MSELoss().to(self.device)

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
            self.model.load_state_dict(checkpoint['G_Net'])
            print('--完成权重加载:{}--'.format(self.load_dir))

    # 训练过程中相关操作
    def train_step(self, data):
        # set_train_data
        self.sdr = data['sdr'].to(self.device)
        self.target = data['target'].to(self.device)

        # cal_loss
        self.hdr = self.model(self.sdr)
        self.loss = self.MSE(self.hdr, self.target)
        self.psnr = kornia.losses.psnr_loss(self.hdr, self.target, max_val=1)

        self.optim_hdr.zero_grad()  # 梯度置零
        self.loss.backward()
        self.optim_hdr.step()  # 更新参数

    def get_current_lr(self):
        optimizers = [self.optim_hdr]
        return [param_group['lr'] for param_group in optimizers[0].param_groups]

    def schedulers_step(self):
        self.sche_hdr.step()

    def save_network(self, epoch):
        mkdir(self.save_dir)
        save_path = self.save_dir + 'G_model_{}.pth'.format(self.vname)
        state = {'G_Net': self.model.state_dict(),
                 'optim_hdr': self.optim_hdr.state_dict(),
                 'epoch': epoch}
        torch.save(state, save_path)

    def tensorboard(self):
        self.tb_step += 1
        loss = self.loss.item()
        psnr = self.psnr.item()
        self.tb.add_scalars('loss', {'loss': loss, 'psnr': psnr}, self.tb_step)
        # self.tb.add_images('sdr', img_tensor=self.sdr[0:self.tbnum], global_step=self.tb_step)
        # self.tb.add_images('hdr', img_tensor=self.hdr[0:self.tbnum], global_step=self.tb_step)
        # self.tb.add_images('gt', img_tensor=self.target[0:self.tbnum], global_step=self.tb_step)
        return loss, psnr
    
    def record_Gnet_para(self):
        para = self.model.state_dict()
        f1 = para['conv1.weight'].view(-1)
        f2 = para['conv2.weight'].view(-1)
        f3 = para['conv3.weight'].view(-1)
        f = torch.cat((f1,f2), 0)
        f = (torch.cat((f, f3), 0)).cpu().numpy()
        F = open(self.G_txt, 'w')
        F.write(str(list(f))[1:-1])
        F.write('\n')
        
class Net_featureExact_D:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids  # 所有gpu号
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        self.lr = opt.lr
        self.gamma = opt.gamma
        self.milestones = opt.milestones
        self.vname = opt.train_hdr.split('/')[-1]
        self.save_dir = opt.save_dir
        # self.load_dir_G = self.save_dir + 'G_model/G_model_{}.pth'.format(self.vname)
        self.load_dir_G = opt.load_dir_G
        self.load_dir_D =  opt.load_dir_D
        self.save_dir = opt.save_dir + 'ITM_model'
        self.G_base_nf = opt.G_base_nf
        self.D_base_nf = opt.D_base_nf
        self.train_status()
        self.tb = SummaryWriter(opt.logdir+'D')
        self.tb_step = 0
        self.tbnum = opt.tbnum
        

    def train_status(self):
        self.model = ITM_net(self.G_base_nf, self.D_base_nf).to(self.device)  # 模型迁移到第一个GPU
        self.set_loss_optimizer_scheduler()
        self.load_network()
        self.model = self.model.to(self.device)
        self.model.train()

    def set_loss_optimizer_scheduler(self):
        self.optim_hdr = optim.Adam(self.model.parameters(), lr=self.lr)
        self.sche_hdr = lr_scheduler.MultiStepLR(self.optim_hdr, milestones=self.milestones, gamma=self.gamma)
        self.MSE = nn.L1Loss().to(self.device)

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

    def load_network(self):     #改为调整后的Gnet参数
        self.init_weight(self.model, 'xavier')
        if self.load_dir_G is not None:
            checkpoint_G = torch.load(self.load_dir_G, map_location=self.device)
            self.model.Gnet.load_state_dict(checkpoint_G['G_Net'])
            print('--完成权重加载:{}--'.format(self.load_dir_G))
        if self.load_dir_D is not None:
            checkpoint_D = torch.load(self.load_dir_D, map_location=self.device)
            self.model.Dnet.load_state_dict(checkpoint_D['D_Net'])
            print('--完成权重加载:{}--'.format(self.load_dir_D))

    # 训练过程中相关操作
    def train_step(self, data):
        # set_train_data
        self.sdr = data['sdr'].to(self.device)
        self.target = data['target'].to(self.device)

        # cal_loss
        self.hdr, self.hdr_G = self.model(self.sdr)
        self.loss = self.MSE(self.hdr, self.target)
        self.psnr = kornia.losses.psnr_loss(self.hdr, self.target, max_val=1)
        self.psnr_G = kornia.losses.psnr_loss(self.hdr_G, self.target, max_val=1)

        self.optim_hdr.zero_grad()  # 梯度置零
        self.loss.backward()
        self.optim_hdr.step()  # 更新参数

    def get_current_lr(self):
        optimizers = [self.optim_hdr]
        return [param_group['lr'] for param_group in optimizers[0].param_groups]

    def schedulers_step(self):
        self.sche_hdr.step()
    
    def save_network(self, epoch):
        mkdir(self.save_dir)
        save_path = self.save_dir + '/ITM_model_{}.pth'.format(self.vname)
        state = {'ITM_Net': self.model.state_dict(),
                 'optim_hdr': self.optim_hdr.state_dict(),
                 'epoch': epoch}
        torch.save(state, save_path)

    def tensorboard(self):
        self.tb_step += 1
        loss = self.loss.item()
        psnr = self.psnr.item()
        psnr_G = self.psnr_G.item()
        self.tb.add_scalars('loss', {'loss': loss, 'psnr': psnr}, self.tb_step)
        return loss, psnr, psnr_G

def featureExact(opt):
    torch.manual_seed(901)
    train_loader = DataLoader(finetune_Dataset(opt), batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    batch_num = len(train_loader)
    # model1 = Net_featureExact_G(opt)
    
    # for epoch in range(opt.epoch_start, opt.epoch_end + 1):
    #     losses = []
    #     psnres = []
    #     lr = model1.get_current_lr()
    #     for i, data in enumerate(train_loader, 1):
    #         model1.train_step(data)
    #         loss, psnr = model1.tensorboard()
    #         losses.append(loss)
    #         psnres.append(psnr)
    #         if i % 10 == 0:
    #             print('--Gnetfinetune_{} epoch:{}, batch:{}/{}, lr:{}, loss:{}, psnr:{}\n'.format(model1.vname, epoch, i, batch_num, lr, loss, psnr))

    #     epoch_message = '--Gnetfinetune_{} epoch:{}, batch_size:{}, lr:{}, loss:{}, psnr:{}'.format(model1.vname, epoch, opt.batch_size, lr, np.mean(losses), np.mean(psnres))
    #     print(epoch_message)
    #     print('------------')
    #     with open(opt.loss_file, 'a', encoding='utf-8') as f:
    #         f.write(epoch_message)
    #         f.write('\n')
    #     model1.schedulers_step()

    # model1.save_network(epoch=epoch)
    # model1.tb.close()
    # model1.record_Gnet_para()

    model2 = Net_featureExact_D(opt)
    for epoch in range(opt.epoch_start, opt.epoch_end + 1):
        losses = []
        psnres = []
        psnres_G = []
        lr = model2.get_current_lr()
        for i, data in enumerate(train_loader, 1):
            model2.train_step(data)
            loss, psnr, psnr_G = model2.tensorboard()
            losses.append(loss)
            psnres.append(psnr)
            psnres_G.append(psnr_G)
            if i % 10 == 0:
                print('--Dnetfinetune_{} epoch:{}, batch:{}/{}, lr:{}, loss:{}, psnr:{}, psnr_G:{}\n'.format(model2.vname, epoch, i, batch_num, lr, loss, psnr, psnr_G))

        epoch_message = '--Dnetfinetune_{} epoch:{}, batch_size:{}, lr:{}, loss:{}, psnr:{}, psnr_G:{}'.format(model2.vname, epoch, 
                                                                                                            opt.batch_size, lr, 
                                                                                                            np.mean(losses),np.mean(psnres),
                                                                                                            np.mean(psnres_G))
        print(epoch_message)
        print('------------')
        with open(opt.loss_file, 'a', encoding='utf-8') as f:
            f.write(epoch_message)
            f.write('\n')
        model2.schedulers_step()
        
    model2.save_network(epoch=epoch) 
    model2.tb.close()
            
