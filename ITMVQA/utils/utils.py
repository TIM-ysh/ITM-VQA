from tqdm import tqdm
import torch
import cv2
import numpy as np
import os
import natsort
np.set_printoptions(suppress=True)

def EOTF_PQ_cuda(ERGB):
    try:
        ERGB = torch.from_numpy(ERGB)  
    except: pass
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ERGB = torch.clamp(ERGB,min=0,max=1).cuda()

    X1 = ERGB ** (1 / m2)
    X2 = X1 - c1
    X2[X2 < 0] = 0

    X3 = c2 - c3 * X1

    X4 = (X2 / X3) ** (1 / m1)
    return X4 * 10000

def EOTF_PQ_cuda_inverse(LRGB):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    RGB_l = LRGB / 10000
    RGB_l = torch.clamp(RGB_l,min=0, max=1).cuda()

    X1 = c1 + c2 * RGB_l ** m1
    X2 = 1 + c3 * RGB_l ** m1
    X3 = (X1 / X2) ** m2
    return X3

def HDR_to_ICTCP(ERGB):
    LRGB = EOTF_PQ_cuda(ERGB)  # hw3
    LR, LG, LB = torch.split(LRGB, 1, dim=-1)  # hw1

    L = (1688 * LR + 2146 * LG + 262 * LB) / 4096
    M = (683 * LR + 2951 * LG + 462 * LB) / 4096
    S = (99 * LR + 309 * LG + 3688 * LB) / 4096
    LMS = torch.cat([L, M, S], dim=-1)  # hw3

    ELMS = EOTF_PQ_cuda_inverse(LMS)  # hw3

    EL, EM, ES = torch.split(ELMS, 1, dim=-1)  # hw1
    I = (2048 * EL + 2048 * EM + 0 * ES) / 4096
    T = (6610 * EL - 13613 * EM + 7003 * ES) / 4096
    P = (17933 * EL - 17390 * EM - 543 * ES) / 4096

    ITP = torch.cat([I,T,P], dim=-1)  # hw3
    return ITP

def ESDR709_to_LSDR2020(ERGB709): #输入为归一化
    LRGB = ERGB709 ** 2.4
    LRGB = torch.from_numpy(LRGB).cuda()
    LR, LG, LB = torch.split(LRGB, 1, dim=-1)  # hw1
    LR2020 = 0.6274 * LR + 0.3293 * LG + 0.0433 * LB
    LG2020 = 0.0691 * LR + 0.9195 * LG + 0.0114 * LB
    LB2020 = 0.0164 * LR + 0.0880 * LG + 0.8956 * LB
    LRGB2020 = torch.cat([LR2020, LG2020, LB2020], dim=-1)  # hw3
    return LRGB2020*100

def SDR_to_ICTCP(ERGB):
    LRGB = ESDR709_to_LSDR2020(ERGB).cuda()

    LR, LG, LB = torch.split(LRGB, 1, dim=-1)  # hw1
    L = (1688 * LR + 2146 * LG + 262 * LB) / 4096
    M = (683 * LR + 2951 * LG + 462 * LB) / 4096
    S = (99 * LR + 309 * LG + 3688 * LB) / 4096
    LMS = torch.cat([L, M, S], dim=-1)  # hw3

    ELMS = EOTF_PQ_cuda_inverse(LMS)  # hw3

    EL, EM, ES = torch.split(ELMS, 1, dim=-1)  # hw1
    I = (2048 * EL + 2048 * EM + 0 * ES) / 4096
    T = (6610 * EL - 13613 * EM + 7003 * ES) / 4096
    P = (17933 * EL - 17390 * EM - 543 * ES) / 4096

    ITP = torch.cat([I,T,P], dim=-1)  # hw3
    return ITP

def ICTCP_to_HDR(ITP, dim=-1):
    ITP = torch.from_numpy(ITP)
    I, T, P = torch.split(ITP, 1, dim=dim)  # hw1
    EL = 1 * I + 0.009 * T + 0.111 * P
    EM = 1 * I - 0.009 * T - 0.111 * P
    ES = 1 * I + 0.560 * T - 0.321 * P
    ELMS = torch.cat([EL, EM, ES], dim=dim)  # hw3

    LMS = EOTF_PQ_cuda(ELMS)
    L, M, S = torch.split(LMS, 1, dim=dim)  # hw1

    X = 2.071 * L - 1.327 * M + 0.207 * S
    Y = 0.365 * L + 0.681 * M - 0.045 * S
    Z = -0.049 * L - 0.05 * M + 1.188 * S

    R = 1.7176 * X - 0.3557 * Y - 0.2534 * Z
    G = -0.6667 * X + 1.6165 * Y + 0.0158 * Z
    B = 0.0176 * X - 0.0428 * Y + 0.9421 * Z

    RGB = torch.cat([R, G, B], dim=dim)  # hw3
    ERGB = EOTF_PQ_cuda_inverse(RGB)
    ERGB = ERGB.cpu().numpy()
    return ERGB

def ICTCP_to_ICH(ITP, dim=-1):
    I, T, P = np.split(ITP, 1, dim=dim)
    H = np.atan(P / T)
    C = (T ** 2 + P ** 2) ** (1 / 2)
    return I,C,H

def BT709_rgb_to_yuv(ERGB):
    ER, EG, EB = cv2.split(ERGB)
    EY = np.clip(0.2126 * ER + 0.7152 * EG + 0.0722 * EB,a_min=0,a_max=1) #跟cv2直接转灰度图公式不同
    EU = (EB - EY) / 1.8556
    EV = (ER - EY) / 1.5748
    EYUV = cv2.merge([EY, EU, EV])
    return EYUV

def BT2020_rgb_to_yuv(ERGB):
    ER, EG, EB = cv2.split(ERGB)
    EY = np.clip(0.2627 * ER + 0.6780 * EG + 0.0593 * EB,a_min=0,a_max=1)
    EU = (EB - EY) / 1.8814
    EV = (ER - EY) / 1.4746
    EYUV = cv2.merge([EY, EU, EV])
    return EYUV

def get_finetune_data(hpath, spath):
    v_list = natsort.natsorted(os.listdir(hpath))
    for v in v_list:
        hv = hpath + '/' + v
        sv = spath + '/' + v
        h_m_list = natsort.natsorted(os.listdir(hv))  #让expand放后面了
        for h_m in h_m_list:
            h_png_path = hv + '/' + h_m
            s_png_path = sv + '/' + h_m.split('_')[-1]
    return h_png_path, s_png_path


def read_Feat(path):
    f = open(path, 'r')
    F = f.read()[:-1].split('\n')   
    feat = []
    for i in range(len(F)):
        feat.append(F[i].split(',')[:])
    f.close()
    feat = np.array(feat, np.float16)
    # feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat)) 
    return feat

def mean_var_muti():       #提取k不定的特征
    for k in range(1,26,2):
        mode = 'G'
        path = []
        if mode == 'G':
            size = [200, 422, k]
        else: size = [200, 51, k]
        outPath1 = 'feat/feat_mean/{}_feat/{}.txt'.format(mode, mode+'_k_frame_A_'+str(k))
        outPath2 = 'feat/feat_delta/{}_feat/{}.txt'.format(mode, mode+'_k_frame_V_'+str(k))
        feat_map = np.empty(size, np.float32)
        for k_frame in range(k):
            path = 'feat/G_16_para_k_1_to25/feat_{}/{}.txt'.format(str(k_frame+1), mode)
            feat = read_Feat(path)
            feat_map[...,k_frame] = feat
        feat_mean = normal_data(np.mean(feat_map, -1)).tolist()
        feat_var = normal_data(np.var(feat_map, -1)).tolist()
        # feat_mean = np.mean(feat_map, -1).tolist()
        # feat_var = np.var(feat_map, -1).tolist()
        # print(len(feat_map.tolist()), feat_map.tolist()[0])

        # feat_mean = (feat_mean - np.min(feat_mean)) / (np.max(feat_mean) - np.min(feat_mean))
        # feat_var = (feat_var - np.min(feat_var)) / (np.max(feat_var) - np.min(feat_var))
        F1 = open(outPath1, 'w')
        F2 = open(outPath2, 'w')
        for i in tqdm(range(200)):
            F1.write(str(feat_mean[i])[1:-1])
            if i != 200-1:
                F1.write('\n')
            F2.write(str(feat_var[i])[1:-1])
            if i != 200-1:
                F2.write('\n')
        F1.close()
        F2.close()

def mean_var():       #提取k定的特征,G_para
    k = 7
    mode = 'G'
    path = []
    G_nf = 16
    for D_nf in [8, 32]:
        G_para_length = 10*G_nf+G_nf*G_nf+6
        D_para_length = D_nf*3+3
        if mode == 'G':
            size = [200, G_para_length, k]
        else: size = [200, D_para_length, k]
        
        outPath1 = 'feat/feat_mean/D_para/{}_feat/{}.txt'.format(mode, mode+'_D_nf_'+str(D_nf)+'_k_frame_A_'+str(k))
        outPath2 = 'feat/feat_delta/D_para/{}_feat/{}.txt'.format(mode, mode+'_D_nf_'+str(D_nf)+'_k_frame_V_'+str(k))
        feat_map = np.empty(size, np.float32)
        for k_frame in range(k):
            path = 'feat/D_para/feat_No_{}_Gnf_16_Dnf_{}/{}.txt'.format(str(k_frame+1), D_nf, mode)
            feat = read_Feat(path)
            feat_map[...,k_frame] = feat
        feat_mean = normal_data(np.mean(feat_map, -1)).tolist()
        feat_var = normal_data(np.var(feat_map, -1)).tolist()

        F1 = open(outPath1, 'w')
        F2 = open(outPath2, 'w')
        for i in tqdm(range(200)):
            F1.write(str(feat_mean[i])[1:-1])
            if i != 200-1: 
                F1.write('\n')
            F2.write(str(feat_var[i])[1:-1])
            if i != 200-1:
                F2.write('\n')
        F1.close()
        F2.close()

def normal_data(data):
    for i in range(200):
        data_normal = (data - np.min(data, 0))/(np.max(data, 0) - np.min(data, 0) + 1e-9)
    return np.around(data_normal, 4)

if __name__ == '__main__':
    
    # path1 = r'E:\NSS\testframes\firstFrame\P-ITM\Art_012.png'
    # path2 = r'E:\NSS\testframes\firstFrame\ITMA\Art_012.png'
    
    # divPic = DivHDR3d(path1, path2)
    # print(divPic.max(), divPic.min())
    
    # divPic = cv2.resize(divPic, (960,540))
    # # cv2.imwrite(r'E:\TIM\HDR\ITM_VQA\testpic\subPic.png', divPic*255)
    # cv2.imshow('img', divPic)
    # cv2.waitKey()
    mean_var()

    # a1 = np.array([[1,2,3], [2,3,4]])
    # a2 = np.array([[5,3,4], [3,4,5]])
    # a = np.dstack([a1, a2])
    # print(a.shape,'\n', a)
    # print(a1)
    # print(np.mean(a, -1))
    # print(np.var(a, -1))
    # print(a.tolist())