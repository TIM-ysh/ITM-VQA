import torchvision as p
import torch.nn as nn
import torch

model = p.models.resnet50(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class FCN(nn.Module):
    def __init__(self):
        super(FCN,self).__init__()
        self.layer1 = nn.Conv2d(256,nClasses,1,stride=1,padding=0,bias=True)
        self.trans = nn.ConvTranspose2d(nClasses,nClasses,2,stride=2,padding=0,bias=True)
        self.layer2 = nn.Conv2d(128,nClasses,1,stride=1,padding=0,bias=True)
        self.up = nn.ConvTranspose2d(nClasses,nClasses,8,stride=8,padding=0,bias=True)
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                #m.weight.detach().normal_(0,0.01)
                nn.init.xavier_uniform(m.weight.data)
                m.bias.detach().zero_()
    def forward(self,x,model):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x1 = model.layer2(x)
        x2 = model.layer3(x1)
        #layers.append(x)#20
        x = model.layer4(x2)
        x = model.avgpool(x)#20
        skip = self.layer1(x2)
        y = skip + x
        c = self.trans(y)
        #### 40
        v = self.layer2(x1)
        y = c+v
        x = self.up(y)
        return x

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
 
    val_acc_history = []
 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
 
        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
 
            running_loss = 0.0
            running_corrects = 0
 
            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                # 零参数梯度
                optimizer.zero_grad()
 
                # 前向
                # 如果只在训练时则跟踪轨迹
                with torch.set_grad_enabled(phase == 'train'):
                    # 获取模型输出并计算损失
                    # 开始的特殊情况，因为在训练中它有一个辅助输出。
                    # 在训练模式下，我们通过将最终输出和辅助输出相加来计算损耗
                    # 但在测试中我们只考虑最终输出。
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
 
                    _, preds = torch.max(outputs, 1)
 
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
 
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
 
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
 
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
 
        print()
 
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
 
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


print(model)