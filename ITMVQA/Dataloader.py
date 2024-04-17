import torch
import re
import numpy as np
import cv2
import os
import torch.utils.data as data
import time
import natsort

def readcsv(csvPath):
    f = open(csvPath, 'r')
    F = re.split('[ \n]', f.read())  #多个字符切片
    mos = {}
    for i in range(200):
        mos[F[2*i]] = float(F[2*i+1])
    f.close()
    return mos

class train_Dataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        sdr_dir = opt.train_sdr
        hdr_dir = opt.train_hdr
        self.sdr_list = []
        self.hdr_list = []
        name_list = os.listdir(hdr_dir)
        for name in name_list:
            self.sdr_list.append(sdr_dir + '/' + name)
            self.hdr_list.append(hdr_dir + '/' + name)

    def __getitem__(self, index):
        input_ = cv2.imread(self.sdr_list[index], flags=-1)[:, :, ::-1] / 255 #处理为全局
        target_ = cv2.imread(self.hdr_list[index], flags=-1)[:, :, ::-1] / 65535

        input_ = np.array(input_, np.float32)
        target_ = np.array(target_, np.float32)

        input_ = torch.from_numpy(input_).float().permute(2, 0, 1)
        target_ = torch.from_numpy(target_).float().permute(2, 0, 1)
        return {'sdr': input_, 'target': target_}

    def __len__(self):
        return len(self.sdr_list)
    
class finetune_Dataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        sdr_dir = opt.train_sdr    
        hdr_dir = opt.train_hdr
        self.sdr_list = []
        self.hdr_list = []
        name = natsort.natsorted(os.listdir(hdr_dir))[opt.k_frame-1]
        sdr_name = name.split('_')[1] + '_' + str(opt.k_frame) + '.' + name.split('.')[-1]
        hdr_name = name
        self.sdr_list.append(sdr_dir + '/' + sdr_name)
        self.hdr_list.append(hdr_dir + '/' + hdr_name)

    def __getitem__(self, index):
        input_ = cv2.imread(self.sdr_list[index], flags=-1)[:, :, ::-1] / 255 #处理为全局
        target_ = cv2.imread(self.hdr_list[index], flags=-1)[:, :, ::-1] / 65535

        input_ = np.array(input_, np.float32)
        target_ = np.array(target_, np.float32)

        input_ = torch.from_numpy(input_).float().permute(2, 0, 1)
        target_ = torch.from_numpy(target_).float().permute(2, 0, 1)
        return {'sdr': input_, 'target': target_}

    def __len__(self):
        return len(self.sdr_list)
    
class ITMVQA_train(data.Dataset):
    def __init__(self, link, opt):
        super().__init__()
        self.mos = self.link_treat(self.readmos(opt, 'itmv'), link)
        self.mos_sdr = self.link_treat(self.readmos(opt, 'sdr'), link)
        self.G_feat_A = self.link_treat(self.read_Feat(opt.G_feat_A), link)
        self.D_feat_A = self.link_treat(self.read_Feat(opt.D_feat_A), link)
        self.G_feat_V = self.link_treat(self.read_Feat(opt.G_feat_V), link)
        self.D_feat_V = self.link_treat(self.read_Feat(opt.D_feat_V), link)
        # print(len(self.G_feat_A), self.G_feat_A[0].shape)
        
    def __getitem__(self, idx):
        # print(idx, self.G_feat_A[idx].shape)
        # print(idx, self.mos[idx], self.mos_sdr[idx])
        G_feat_A = torch.from_numpy(self.G_feat_A[idx])
        D_feat_A = torch.from_numpy(self.D_feat_A[idx])
        G_feat_V = torch.from_numpy(self.G_feat_V[idx])
        D_feat_V = torch.from_numpy(self.D_feat_V[idx])
        mos = torch.from_numpy(np.array(self.mos[idx], np.float32))
        mos_sdr = torch.from_numpy(np.array(self.mos_sdr[idx], np.float32))

        return {'mos':mos, 'mos_sdr':mos_sdr, 'G_feat_A':G_feat_A, 'D_feat_A':D_feat_A, 'G_feat_V':G_feat_V, 'D_feat_V':D_feat_V}

    def __len__(self):
        return len(self.mos)

    def readmos(self, opt, mos_mode):
        if mos_mode == 'itmv':
            csvPath = opt.mos
        elif mos_mode == 'sdr':
            csvPath = opt.mos_sdr
        f = open(csvPath, 'r')
        F = f.read().split('\n')
        mos = []
        for i in range(200):
            mos.append(F[i])
        f.close()
        mos = np.array(mos, np.float32)
        return mos
    
    def read_Feat(self, path):
        f = open(path, 'r')
        F = f.read()[:-1].split('\n')
        feat = []
        for i in range(len(F)):
            feat.append(F[i].split(',')[:])
        f.close()
        return np.array(feat, np.float32)
    
    def link_treat(self, feat, link):
        F = []
        for i in link:
            for j in range(10*i, 10*i+10):
                F.append(feat[j])
        return F

if __name__ == '__main__':
    path = 'E:/TIM/HDR/ITM_VQA/SVM/MOS/MOS.csv'
    
    sdrpng = 'F:/lzj/sdr/009.png'
    hdrpng = 'F:/lzj/hdr/009.png'
    
    t1 = time.time()
    
    model = decurve(sdrpng, hdrpng)
    img = model.PredimgITP()
    # cv2.imshow('img', img[:,:,::-1]/1023.0)
    # cv2.waitKey()
    print(img.shape)
    print('time: ', time.time()-t1)
    # mos = readcsv(path)
    # print(mos)