import torch
import torch.nn as nn
import torch.nn.functional as F


class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 1
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, 3, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.cond_feature = []

    def forward(self, x):
        self.cond_feature = []
        conv1_out = self.relu(self.conv1(x))   #torch.Size([4, 32, 240, 240]) 
        conv2_out = self.relu(self.conv2(conv1_out))   #torch.Size([4, 32, 120, 120])
        conv3_out = self.relu(self.conv3(conv2_out))  #torch.Size([4, 32, 60, 60])
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)  #torch.Size([4, 32])
        self.cond_feature = self.cond_feature + [out]
        return out

# 3layers with control
class G_model(nn.Module):
    def __init__(self, base_nf, in_nc=3, out_nc=3, cond_nf=32):
        super(G_model, self).__init__()
        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_scale2 = nn.Linear(cond_nf, base_nf,  bias=True)
        self.cond_scale3 = nn.Linear(cond_nf, 3, bias=True)

        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, 3, bias=True)

        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True) 
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = self.cond_net(x)
        self.C_feat = cond
        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        out = self.conv1(x)
        out = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)
        
        out = self.conv2(out)
        out = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv3(out)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out
        self.CML_feat = torch.cat([scale1.view(-1, self.base_nf), scale2.view(-1, self.base_nf), scale3.view(-1, self.out_nc),
                                   shift1.view(-1, self.base_nf), shift2.view(-1, self.base_nf), shift3.view(-1, self.out_nc)], 1)
        return out

class residual_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Conv2d(in_c,out_c,3,1,1)
        self.conv2 = nn.Conv2d(out_c,out_c,3,1,1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, input):
        x1 = self.act(self.bn1(self.conv1(input)))
        x2 = self.bn2(self.conv2(x1))
        if self.in_c==self.out_c:
            output = self.act(input + x2)
        else:
            output = self.act(x2)
        return output

class D_model(nn.Module):
    def __init__(self, base_nf, in_c=3, out_c=3, cond_nf=32):
        super().__init__()
        self.base_nf = base_nf
        self.out_c = out_c
        self.cond = Condition()
        self.res1 = residual_block(in_c, base_nf)
        self.res2 = residual_block(base_nf, base_nf)
        self.res3 = residual_block(base_nf, base_nf)
        self.conv = nn.Conv2d(base_nf, out_c, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.gp = nn.AdaptiveAvgPool2d((1,1))

        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_scale2 = nn.Linear(cond_nf, base_nf,  bias=True)
        self.cond_scale3 = nn.Linear(cond_nf, base_nf,  bias=True)
        self.cond_scale4 = nn.Linear(cond_nf, out_c, bias=True)

        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift4 = nn.Linear(cond_nf, out_c, bias=True)
        
    def forward(self, input):
        cond = self.cond(input[1])
        self.C_feat = cond
        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        scale4 = self.cond_scale4(cond)
        shift4 = self.cond_shift4(cond)
        
        out = self.res1(input[0])
        out1 = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out1)
        
        out = self.res2(out)
        out2 = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out2)

        out = self.res3(out)
        out3 = out * scale3.view(-1, self.base_nf, 1, 1) + shift3.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out3)

        out = self.conv(out)
        out4 = out * scale4.view(-1, self.out_c, 1, 1) + shift4.view(-1, self.out_c, 1, 1) + out

        self.D_feat = torch.cat([self.gp(out1), self.gp(out2), self.gp(out3), self.gp(out4)], 1)
        return out4+input[0]


class ITM_net(nn.Module):
    def __init__(self, G_base_nf=16, D_base_nf=16):
        super().__init__()
        self.Gnet = G_model(G_base_nf)
        self.Dnet = D_model(D_base_nf)
        
    def forward(self, x):
        x1 = self.Gnet(x).detach()
        output = self.Dnet([x1, x])
        return output, x1

class IQA_net_GDGTDT(nn.Module):
    def __init__(self, G_n, D_n):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear((G_n+D_n)*2,128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(G_n)
        self.bn2 = nn.BatchNorm1d(G_n)
        self.bn3 = nn.BatchNorm1d(D_n)
        self.bn4 = nn.BatchNorm1d(D_n)

    def forward(self, input):
        G_feat_A = self.bn1(input['G_feat_A'])
        D_feat_A = self.bn3(input['D_feat_A'])
        G_feat_V = self.bn2(input['G_feat_V'])
        D_feat_V = self.bn4(input['D_feat_V'])
        G_feat_A = input['G_feat_A']
        D_feat_A = input['D_feat_A']
        G_feat_V = input['G_feat_V']
        D_feat_V = input['D_feat_V']
        feat = torch.cat([G_feat_A, D_feat_A, G_feat_V, D_feat_V], 1)
        feat = self.relu(feat)
        x = F.dropout(feat, 0.1)
        # x = feat
        pred = self.fc1(x)#*input['mos_sdr'].unsqueeze(-1)
        # print(pred.shape)
        # print(input['mos_sdr'][:,0].shape)
        # print((self.fc1(x)*input['mos_sdr'][:,0].unsqueeze(-1)).shape)
        pred = self.relu(pred)
        pred = self.fc2(pred)*input['mos_sdr'].unsqueeze(-1)
        return pred.squeeze(-1)
    
class IQA_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(946,128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(422)
        self.bn2 = nn.BatchNorm1d(422)
        self.bn3 = nn.BatchNorm1d(51)
        self.bn4 = nn.BatchNorm1d(51)

    def forward(self, input):
        G_feat_A = self.bn1(input['G_feat_A'])
        D_feat_A = self.bn3(input['D_feat_A'])
        G_feat_V = self.bn2(input['G_feat_V'])
        D_feat_V = self.bn4(input['D_feat_V'])
        G_feat_A = input['G_feat_A']
        D_feat_A = input['D_feat_A']
        G_feat_V = input['G_feat_V']
        D_feat_V = input['D_feat_V']
        feat = torch.cat([G_feat_A, D_feat_A, G_feat_V, D_feat_V], 1)
        feat = self.relu(feat)
        x = F.dropout(feat, 0.1)
        # x = feat
        pred = self.fc1(x)#*input['mos_sdr'].unsqueeze(-1)
        # print(pred.shape)
        # print(input['mos_sdr'][:,0].shape)
        # print((self.fc1(x)*input['mos_sdr'][:,0].unsqueeze(-1)).shape)
        pred = self.relu(pred)
        pred = self.fc2(pred)*input['mos_sdr'].unsqueeze(-1)
        return pred.squeeze(-1)
    
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

# class vgg_net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.vgg = torchvision.models.vgg16(pretrained=True)
#         self.vgg.eval()
        
#     def forward(self, input):
#         return self.vgg(input)

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = torch.rand((4, 3, 480, 480)).to(device)
    
    # model = G_model().to(device).train()
    
    # para = model.state_dict()
    # # print(para.keys())

    # outputs = model(inputs)
    # print(model.CML_feat)
    # print(outputs.size())
    # net = ITM_net()
    net = IQA_net()

    # 计算网络可学习的参数量
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("网络可学习的参数量为：", num_params)
