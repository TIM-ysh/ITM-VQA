import torch
import numpy as np
from Dataloader import finetune_Dataset
from torch.utils.data import DataLoader
from torch.nn import init
from model import ITM_net
import argparse
import natsort
import os
import cv2

np.set_printoptions(suppress=True)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class Net:
    def __init__(self, opt):
        self.gpu_ids = opt.gpu_ids  # 所有gpu号
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 第一个GPU
        self.load_dir = opt.load_dir
        self.G_base_nf = opt.G_base_nf
        self.D_base_nf = opt.D_base_nf
        self.test_status()
        self.tb_step = 0

    def test_status(self):
        self.model = ITM_net(self.G_base_nf, self.D_base_nf).to(self.device)
        self.load_network()
        self.model.eval()

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

    def test_step(self, data):
        # set_train_data
        with torch.no_grad():
            self.sdr = data['sdr'].to(self.device)
            self.hdr,_ = self.model(self.sdr)
            # self.H = self.sdr.size(2)
            # self.W = self.sdr.size(3)

    def save_para(self):
        return 0

    def Gnet_para(self):
        para = self.model.Gnet.state_dict()
        f1 = para['conv1.weight'].view(-1)
        f2 = para['conv2.weight'].view(-1)
        f3 = para['conv3.weight'].view(-1)
        f = torch.cat([f1, f2, f3], 0)
        return f
        
def test(opt):
    torch.manual_seed(901)
    test_loader = DataLoader(finetune_Dataset(opt), shuffle=False)
    model = Net(opt)
    F_G = open(opt.G_txt, 'a')
    F_D = open(opt.D_txt, 'a')
        
    for i, data in enumerate(test_loader, 1):
        model.test_step(data)
        G_feat_1 = model.model.Gnet.CML_feat
        G_feat_2 = model.Gnet_para()
        G_feat = torch.cat([G_feat_1.view(-1), G_feat_2.view(-1)], 0)
        D_feat = model.model.Dnet.D_feat   #torch.Size([1, 51, 1, 1])

        F_G.write(str(list(G_feat.view(-1).cpu().numpy()))[1:-1])
        F_G.write('\n')
        F_D.write(str(list(D_feat.view(-1).cpu().numpy()))[1:-1])
        F_D.write('\n')
    F_D.close()
    F_G.close()

def ITMVQA_para(): #特征提取
    parser = argparse.ArgumentParser(description='ITMVQA_para')
    opt = parser.parse_args()
    opt.gpu_ids = [0]
    opt.num_workers = 0
    opt.mode = 'D'
    opt.hpath = 'I:/ITMVQADATA/ITMVQADataset_hdr'
    opt.spath = 'I:/ITMVQADATA/ITMVQADataset_sdr'
    v_list = natsort.natsorted(os.listdir(opt.hpath))
    for nf in [8, 32]:
        opt.G_base_nf = 16
        opt.D_base_nf = nf
        for k in range(1, 8):
            opt.k_frame = k
            opt.G_txt = 'feat/D_para/feat_No_{}_Gnf_16_Dnf_{}/G.txt'.format(str(opt.k_frame), str(nf))
            opt.D_txt = 'feat/D_para/feat_No_{}_Gnf_16_Dnf_{}/D.txt'.format(str(opt.k_frame), str(nf))
            mkdir(opt.G_txt.split('/')[0] + '/' + opt.G_txt.split('/')[1])
            mkdir(opt.G_txt.split('/')[0] + '/' + opt.G_txt.split('/')[1] + '/' + opt.G_txt.split('/')[2])
            mkdir(opt.D_txt.split('/')[0] + '/' + opt.D_txt.split('/')[1])
            mkdir(opt.D_txt.split('/')[0] + '/' + opt.D_txt.split('/')[1] + '/' + opt.D_txt.split('/')[2])
            for v in v_list:
                hv = opt.hpath + '/' + v
                sv = opt.spath + '/' + v
                h_m_list = natsort.natsorted(os.listdir(hv))  #让expand放后面了
                for h_m in h_m_list:
                    opt.train_hdr = hv + '/' + h_m
                    opt.train_sdr = sv + '/' + h_m.split('_')[-1]
                    opt.load_dir = 'saved_model/D_para/saved_fineture_model_{}_basenf_{}_k{}/ITM_model/ITM_model_'.format(opt.mode, str(nf), str(opt.k_frame)) + h_m + '.pth'
                    test(opt)
    
if __name__ == '__main__':
    ITMVQA_para()