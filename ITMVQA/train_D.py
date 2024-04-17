import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import kornia
import numpy as np
from Dataloader import train_Dataset
from torch.utils.data import DataLoader
from torch.nn import init
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import ITM_net

class Net_D:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids  # 所有gpu号
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        self.lr = opt.lr
        self.gamma = opt.gamma
        self.milestones = opt.milestones
        self.save_dir = opt.save_dir
        self.load_dir = opt.load_dir
        self.base_nf = opt.base_nf
        self.train_status()
        self.tb = SummaryWriter(opt.logdir)
        self.tb_step = 0
        self.tbnum = opt.tbnum

    def train_status(self):
        self.model = ITM_net(base_nf = self.base_nf).to(self.device)  # 模型迁移到第一个GPU
        self.set_loss_optimizer_scheduler()
        self.load_network()
        # if self.gpu_ids:
        #     self.model = nn.DataParallel(self.model, self.gpu_ids)  # 模型迁移到多个GPU上
        self.model = self.model.to(self.device)

        self.model.Dnet.train()
        self.model.Gnet.eval()
        # self.model.Gnet.detach()

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

    def load_network(self):
        self.init_weight(self.model, 'xavier')
        if self.load_dir is not None:
            checkpoint = torch.load(self.load_dir, map_location=self.device)
            self.model.Gnet.load_state_dict(checkpoint['G_Net'])
            print('--完成权重加载:{}--'.format(self.load_dir))

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
        self.loss.requires_grad_(True)
        self.loss.backward()
        self.optim_hdr.step()  # 更新参数

    def get_current_lr(self):
        optimizers = [self.optim_hdr]
        return [param_group['lr'] for param_group in optimizers[0].param_groups]

    def schedulers_step(self):
        self.sche_hdr.step()

    def save_network(self, epoch):
        save_path = self.save_dir + 'model_{}.pth'.format(epoch)
        state = {'D_Net': self.model.Dnet.state_dict(),
                 'G_Net': self.model.Gnet.state_dict(),
                 'optim_hdr': self.optim_hdr.state_dict(),
                 'epoch': epoch}
        torch.save(state, save_path)

    def tensorboard(self):
        self.tb_step += 1

        loss = self.loss.item()
        psnr = self.psnr.item()
        psnr_G = self.psnr_G.item()
        self.tb.add_scalars('loss', {'loss': loss, 'psnr': psnr}, self.tb_step)
        # self.tb.add_images('sdr', img_tensor=self.sdr[0:self.tbnum], global_step=self.tb_step)
        # self.tb.add_images('hdr', img_tensor=self.hdr[0:self.tbnum], global_step=self.tb_step)
        # self.tb.add_images('gt', img_tensor=self.target[0:self.tbnum], global_step=self.tb_step)
        return loss, psnr, psnr_G
        
def train_D(opt):
    torch.manual_seed(901)
    train_loader = DataLoader(train_Dataset(opt), batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    batch_num = len(train_loader)
    model = Net_D(opt)
    
    for epoch in range(opt.epoch_start, opt.epoch_end + 1):
        losses = []
        psnres = []
        psnres_G = []
        lr = model.get_current_lr()
        for i, data in enumerate(train_loader, 1):
            model.train_step(data)
            loss, psnr, psnr_G = model.tensorboard()
            losses.append(loss)
            psnres.append(psnr)
            psnres_G.append(psnr_G)
            if i % 10 == 0:
                print('epoch:{}, batch:{}/{}, lr:{}, loss:{}, psnr:{}, psnr_G:{}\n'.format(epoch, i, batch_num, lr, loss, psnr, psnr_G))

        epoch_message = 'epoch:{}, batch_size:{}, lr:{}, loss:{}, psnr:{}, psnr_G:{}'.format(epoch, opt.batch_size, lr, np.mean(losses), np.mean(psnres), np.mean(psnres_G))
        print(epoch_message)
        print('------------')
        with open(opt.loss_file, 'a', encoding='utf-8') as f:
            f.write(epoch_message)
            f.write('\n')

        model.schedulers_step()
        if epoch % opt.save_epoch == 0:
            model.save_network(epoch=epoch)
    model.tb.close()