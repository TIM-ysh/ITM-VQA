import argparse
import os
import shutil
from train_G import train_G
from train_D import train_D
import natsort
from finetune_featureExact import featureExact
from ITMVQA_demo import ITMVQA_FC

def check_dir(opt):
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.logdir)


def train_param_G(): #预训练
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    
    opt.train_sdr = 'E:/HDRTV_dataset/sdr'
    opt.train_hdr = 'E:/HDRTV_dataset/hdr'
    opt.gpu_ids = [0]
    opt.num_workers = 0
    opt.batch_size = 32
    opt.tbnum = 2

    opt.step = 1
    opt.epoch_start = 1
    opt.epoch_end = 200
    opt.milestones = [40,80,120,160,200]
    opt.save_epoch = 20
    opt.lr = 5e-4
    opt.gamma = 0.5
    opt.load_dir = None

    for i in [8, 32, 64]:
        opt.save_dir = 'saved_model/pred_G_model_'+ str(i) + '/'
        # opt.save_dir = 'saved_model/pred_D_model/'
        opt.loss_file = opt.save_dir+'loss.txt'
        opt.logdir = opt.save_dir+'logs'
        opt.G_base_nf = i      #设置G_net参数量
        check_dir(opt)
        train_G(opt)
    # train_D(opt)

def train_param_D(): #预训练
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    
    opt.train_sdr = 'E:/HDRTV_dataset/sdr'
    opt.train_hdr = 'E:/HDRTV_dataset/hdr'
    opt.gpu_ids = [0]
    opt.num_workers = 0
    opt.batch_size = 32
    opt.tbnum = 2

    opt.step = 1
    opt.epoch_start = 1
    opt.epoch_end = 200
    opt.milestones = [40,80,120,160,200]
    opt.save_epoch = 20
    opt.lr = 5e-4
    opt.gamma = 0.5
    opt.load_dir = None

    for i in [8, 32, 64]:
        opt.save_dir = 'saved_model/pred_G_model_'+ str(i) + '/'
        # opt.save_dir = 'saved_model/pred_D_model/'
        opt.loss_file = opt.save_dir+'loss.txt'
        opt.logdir = opt.save_dir+'logs'
        opt.G_base_nf = i      #设置G_net参数量
        check_dir(opt)
        opt.base_nf = 16
        train_D(opt)

def featureExact_G_param(): #精调并提取特征
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    opt.hpath = 'E:/ITMVQADATA/ITMVQADataset_hdr'
    opt.spath = 'E:/ITMVQADATA/ITMVQADataset_sdr'
    opt.gpu_ids = [0]
    opt.num_workers = 0
    opt.batch_size = 1
    opt.tbnum = 2

    opt.step = 1
    opt.epoch_start = 1
    opt.epoch_end = 80
    opt.milestones = [10, 20, 30, 40, 50, 60, 70, 80]
    opt.save_epoch = 80
    opt.lr = 1e-3
    opt.gamma = 0.5
    opt.D_base_nf = 16
    for i in [32, 64]:
        # opt.save_dir = 'saved_model/pred_G_model_'+ str(i) + '/'
        opt.G_base_nf = i
        opt.load_dir_G = 'saved_model/pred_G_model/pred_G_model_' + str(i) + '/model_200.pth'
        opt.load_dir_D = 'saved_model/pred_D_model/model_100.pth'

        for k in range(1, 8):
            opt.k_frame = k
            opt.save_dir = 'saved_model/saved_fineture_model_G_basenf_' + str(i) + '_k' + str(opt.k_frame) + '/'
            opt.loss_file = opt.save_dir + 'loss.txt'
            opt.logdir = opt.save_dir + 'logs'
            opt.G_txt = opt.save_dir + 'G_para.txt'
            check_dir(opt)
            v_list = natsort.natsorted(os.listdir(opt.hpath))
            for v in v_list:
                hv = opt.hpath + '/' + v
                sv = opt.spath + '/' + v
                h_m_list = natsort.natsorted(os.listdir(hv))  #让expand放后面了
                for h_m in h_m_list:
                    opt.train_hdr = hv + '/' + h_m
                    opt.train_sdr = sv + '/' + h_m.split('_')[-1]
                    featureExact(opt)

def featureExact_D_param(): #精调并提取特征
    parser = argparse.ArgumentParser(description='')
    opt = parser.parse_args()
    opt.hpath = 'I:/ITMVQADATA/ITMVQADataset_hdr'
    opt.spath = 'I:/ITMVQADATA/ITMVQADataset_sdr'
    opt.gpu_ids = [0]
    opt.num_workers = 0
    opt.batch_size = 1
    opt.tbnum = 2

    opt.step = 1
    opt.epoch_start = 1
    opt.epoch_end = 80
    opt.milestones = [10, 20, 30, 40, 50, 60, 70, 80]
    opt.save_epoch = 80
    opt.lr = 1e-3
    opt.gamma = 0.5
    opt.G_base_nf = 16
    for i in [32, 64]:
        # opt.save_dir = 'saved_model/pred_G_model_'+ str(i) + '/'
        opt.D_base_nf = i
        # opt.load_dir_G = 'saved_model/pred_G_model/pred_G_model_16/model_200.pth'
        opt.load_dir_D = 'saved_model/pred_D_model/pred_D_model_' + str(i) + '/model_50.pth'

        for k in range(1, 8):
            opt.k_frame = k
            opt.save_dir = 'saved_model/D_para/saved_fineture_model_D_basenf_' + str(i) + '_k' + str(opt.k_frame) + '/'
            opt.loss_file = opt.save_dir + 'loss.txt'
            opt.logdir = opt.save_dir + 'logs'
            opt.G_txt = opt.save_dir + 'G_para.txt'
            check_dir(opt)
            v_list = natsort.natsorted(os.listdir(opt.hpath))
            for v in v_list:
                hv = opt.hpath + '/' + v
                sv = opt.spath + '/' + v
                h_m_list = natsort.natsorted(os.listdir(hv))  #让expand放后面了
                for h_m in h_m_list:
                    opt.load_dir_G = 'saved_model/G_para/saved_fineture_model_G_basenf_16_k' + str(opt.k_frame) + '/G_model/G_model_' + h_m + '.pth'
                    opt.train_hdr = hv + '/' + h_m
                    opt.train_sdr = sv + '/' + h_m.split('_')[-1]
                    featureExact(opt)
                    
def ITMVQA_G_para(): #G特征维数变化
    parser = argparse.ArgumentParser(description='ITMVQA_para')
    opt = parser.parse_args()
    opt.mos = 'mos/MOS_list.csv'
    opt.mos_sdr = 'mos/MOS_SDR.csv'
    opt.k = '7'
    for G_nf in [8, 16, 32, 64]:
        opt.G_feat_A = 'feat/feat_mean/G_para/G_feat/G_G_nf_' + str(G_nf) + '_k_frame_A_7.txt'
        opt.D_feat_A = 'feat/feat_mean/G_para/D_feat/D_G_nf_' + str(G_nf) + '_k_frame_A_7.txt'
        opt.G_feat_V = 'feat/feat_delta/G_para/G_feat/G_G_nf_' + str(G_nf) + '_k_frame_V_7.txt'
        opt.D_feat_V = 'feat/feat_delta/G_para/D_feat/D_G_nf_' + str(G_nf) + '_k_frame_V_7.txt'
        opt.G_n = 10*G_nf + G_nf*G_nf + 6
        opt.num = 0
        opt.gpu_ids = [0]
        opt.num_workers = 0
        opt.batch_size = 16
        opt.val = True
        opt.tbnum = 2

        opt.step = 1
        opt.epoch_start = 1
        opt.epoch_end = 300
        opt.milestones = [80, 160, 240]
        opt.save_epoch = 300
        opt.lr = 1e-3
        opt.gamma = 0.1

        opt.mode = 'GDGTDT'
        opt.load_dir = None
        opt.save_dir = 'ITMVQANet_index/G_para/G_para_model_G_nf_' + str(G_nf) + '/'
        opt.logdir = opt.save_dir + 'log/'
        opt.loss_file = opt.save_dir + 'loss_file.txt'
        opt.record_pred = opt.save_dir+'pred_recored.txt'
        opt.loss_file_best = opt.save_dir + 'loss_file_best.txt'
        opt.train_test_num = 10
        check_dir(opt)
        ITMVQA_FC(opt)
    
def ITMVQA_D_para(): #G特征维数变化
    parser = argparse.ArgumentParser(description='ITMVQA_para')
    opt = parser.parse_args()
    opt.mos = 'mos/MOS_list.csv'
    opt.mos_sdr = 'mos/MOS_SDR.csv'
    opt.k = '7'
    G_nf = 16
    opt.G_n = 10*G_nf + G_nf*G_nf + 6
    for D_nf in [8, 32]:
        opt.G_feat_A = 'feat/feat_mean/D_para/G_feat/G_D_nf_' + str(D_nf) + '_k_frame_A_7.txt'
        opt.D_feat_A = 'feat/feat_mean/D_para/D_feat/D_D_nf_' + str(D_nf) + '_k_frame_A_7.txt'
        opt.G_feat_V = 'feat/feat_delta/D_para/G_feat/G_D_nf_' + str(D_nf) + '_k_frame_V_7.txt'
        opt.D_feat_V = 'feat/feat_delta/D_para/D_feat/D_D_nf_' + str(D_nf) + '_k_frame_V_7.txt'
        # opt.G_n = 10*G_nf + G_nf*G_nf + 6
        opt.D_n = D_nf*3 + 3
        opt.num = 0
        opt.gpu_ids = [0]
        opt.num_workers = 0
        opt.batch_size = 16
        opt.val = True
        opt.tbnum = 2

        opt.step = 1
        opt.epoch_start = 1
        opt.epoch_end = 300
        opt.milestones = [80, 160, 240]
        opt.save_epoch = 300
        opt.lr = 1e-3
        opt.gamma = 0.1

        opt.mode = 'GDGTDT'
        opt.load_dir = None
        opt.save_dir = 'ITMVQANet_index/D_para/D_para_model_D_nf_' + str(D_nf) + '/'
        opt.logdir = opt.save_dir + 'log/'
        opt.loss_file = opt.save_dir + 'loss_file.txt'
        opt.record_pred = opt.save_dir+'pred_recored.txt'
        opt.loss_file_best = opt.save_dir + 'loss_file_best.txt'
        opt.train_test_num = 10
        check_dir(opt)
        ITMVQA_FC(opt)

if __name__ == '__main__':
    # train_param_G() 
    # train_param_D() 
    # featureExact_G_param()
    # featureExact_D_param()
    ITMVQA_D_para()

