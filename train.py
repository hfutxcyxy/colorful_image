# coding=gbk
from tools.model import Model
from tools.model_improve import improved_model
from tools.model_improve1 import improved_model1
from tools.model_improve2 import improved_model2
from tools.model_improve3 import improved_model3
import torch.nn.utils.prune as prune
from tools.data_generator import Preprocessor, ValDataset #train_gen, valid_gen
from tools.recorder import Loss_recorder
import numpy as np
from tools.SSIM import SSIM
import sys

import pdb
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import epoches, batch_size, learning_rate,beta_1, beta_2, weight_decay

import argparse
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from torchvision import transforms
import torchvision

import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/yuanxueyu/ml-homework/colorful-pytorch-main")
#torch.backends.cudnn.enabled = False
#训练模型
def train_net(args, model, device, train_loader, optimizer, epoch, weight):

    model.train()
    print('start training')
    
    loss_record = [] # record loss in batches
    
    for batch_idx, data in enumerate(tqdm.tqdm(train_loader)):
        
#         data, _ = data # This is for imagenet training dataset
        x, gt = data
        x, gt = x.to(device), gt.to(device)
        optimizer.zero_grad()
        y,_ = model(x)
        # back propogate
        loss = _loss(y, gt, weight)
        if torch.isnan(loss):
            pdb.set_trace()
        train_rec.take(loss / x.shape[0])
        loss.backward()
        optimizer.step()
        
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_rec.save()
            if args.dry_run:
                break
    return

#损失函数
def _loss(y, gt, weight):  
    # calculate loss
    #print(f"y/gt's shape:{y.shape,gt.shape}")
    loss_tmp = - gt * torch.log(y + 1e-10)
    loss_tmp_perpix = torch.sum(loss_tmp, axis = 1)
    max_idx_perpix = torch.argmax(gt, axis = 1) 
    prior_perpix = weight[max_idx_perpix.cpu()]
    prior_perpix = torch.tensor(prior_perpix).to(device)

    loss_perpix = prior_perpix * loss_tmp_perpix
    loss = torch.sum(loss_perpix) / (y.shape[2] * y.shape[3])

    ssim = SSIM()
    ssim_loss = ssim(gt, torch.tensor(y))

    total_loss = loss * 0.17 + ssim_loss * 0.83
    return total_loss

#测试模型   
def test_net(args, model, device, test_loader, weight):
    model.eval()
    test_loss = 0
    print('start validation')
    with torch.no_grad():
        for x, gt in tqdm.tqdm(test_loader):
            x, gt = x.to(device), gt.to(device)
            y, _ = model(x)
            test_loss += _loss(y, gt, weight).item()
            if args.dry_run:
                break
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(
    test_loss))
            
    
    return test_loss

def prune_model(model, pruning_rate):
    # 遍历模型中的每一层
    for name, module in model.named_modules():
        # 如果这一层是卷积层或全连接层
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # 应用L1非结构化剪枝方法，移除权重的一定比例
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
    return model

def quantize_model(model):
    model.eval()
    # 使用全张量的权重观察器
    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8)
    )
    model.qconfig = qconfig
    model = torch.quantization.prepare(model)
    model = torch.quantization.convert(model)
    return model

def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    return model

def train_prune_quantize(args, model, device, train_loader, val_loader, optimizer, weight, pruning_rate, num_pruning_steps, num_finetune_steps):
    for pruning_step in range(num_pruning_steps):
        # 在每个剪枝步骤开始时，对模型进行剪枝
        model = prune_model(model, pruning_rate)

        # 在剪枝后进行一定数量的微调迭代
        for finetune_step in range(num_finetune_steps):
            train_net(args, model, device, train_loader, optimizer, finetune_step, weight)
            loss = test_net(args, model, device, val_loader, weight)

        print('Pruning step {} finished'.format(pruning_step))
        #model = quantize_model(model)

    if args.save_model:
        model = remove_pruning(model)
        torch.save(model.state_dict(), "model_shot3.pt")
        print('model is saved at', 'model.pt')

def assign_weight(prior):
    if prior <= 8.809291375334112e-06:
        return 5.61
    elif prior <= 1.4823259786852056e-05:
        return 5.59  # 对于 prior < 0.1，weight 为 10
    elif prior <= 3.0900533493534945e-05:
        return 5.56
    elif prior <= 4.885578261975041e-05:
        return 5.53
    elif prior <= 6.986586067252938e-05:
        return 5.50
    elif prior <= 0.00012560275218950715:
        return 5.40  # 对于 0.1 <= prior < 0.2，weight 为 8
    elif prior <= 0.0006463244856606789:
        return 4.67  # 对于 0.2 <= prior < 0.3，weight 为 6
    elif prior <= 0.0008272200181830892:
        return 4.46
    elif prior <= 0.001353290586727366:
        return 3.949592506637025
    elif prior <= 0.0016059771065649992:
        return 3.74  # 对于 0.3 <= prior < 0.4，weight 为 4
    elif prior <= 0.0024515419144133617:
        return 3.1813823025721666
    elif prior <= 0.003425008384493324:
        return 2.7135549021896637
    elif prior <= 0.004187121029014661:
        return 2.4334096945217434
    elif prior <= 0.005147817796708375:
        return 2.153192600112245
    elif prior <= 0.00616122326220804:
        return 1.9199699452776036
    elif prior <= 0.00710936814080711:
        return 1.7433041467769297
    elif prior <= 0.007870717010616516:
        return 1.6233592708167377
    else:
        return 1.0213350698576873  # 对于 prior >= 0.4，weight 为 2

train_rec = Loss_recorder('train')
val_rec = Loss_recorder('val')

                
                   
if __name__ == '__main__':
    
    # Add arguments
    parser = argparse.ArgumentParser(description='PyTorch Image Colorization')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=epoches, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=learning_rate, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--smoth-prior',type = str, default = 'data/prior_prob_smoothed.npy', help = 'the path to the smoothed prior distribution')
    parser.add_argument('--parellel',action = 'store_true', default = True, help = 'whether to apply parellecl computing')
    parser.add_argument('--resume',action = 'store_true', default = False, help = 'resume unfinished training')

    args = parser.parse_args()
    
    #选择设备
    # specify device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else "cpu")

    # data loader configs
    train_path = 'data/train'
    val_path = 'data/valid'
    #device_ids = [0,1]
    
    kwargs = {'num_workers': 4 if args.parellel else 1,#args.num_workers,
              'pin_memory': True} if args.parellel else {}
    
    #预处理
    transform = transforms.Compose([Preprocessor(),])
    
    # If you are using imagenet, uncomment this    
    # train_ds = torchvision.datasets.ImageFolder(train_path, 
    #                                             transform=transform)
    
    # If you are using coco, uncomment this
    #加载一个数据集
    train_ds = ValDataset(train_path, 
                        transform=transform)
    #trainloader
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.batch_size * 
                                               torch.cuda.device_count() if args.parellel else args.batch_size,
                                               shuffle=True,
                                               **kwargs)
    
    val_ds = ValDataset(val_path, 
                        transform=transform)
    val_loader = torch.utils.data.DataLoader(val_ds,
                                               batch_size=args.test_batch_size,
                                               shuffle=True,
                                             **kwargs)
    
    
    
    # build model
    model = Model()
    """
    if args.parellel and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model,device_ids=[0,1])
    """
    if args.resume:
        model.load_state_dict(torch.load("model.pt"))

    model.to(device)
    
    #调用adam优化器
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                          betas = (beta_1, beta_2),
                          weight_decay = weight_decay,
                          )
        
    # load prior distribution
    #加载平滑分布先验
    prior = np.load(args.smoth_prior)
    
    # calculate weight
    """
    weight = 1/(0.5 * prior + 0.5 / 313)
    weight = weight / sum(prior * weight)
    """

    # 确保先验概率是一个Tensor
    #第一种方法
    """
    prior = torch.tensor(prior, dtype=torch.float64)

    weight = 5.6 - 184 * prior
    condition1 = prior >= 0.02
    weight = torch.where(condition1, 1.92 - 0.01 * prior, weight)
    weight = weight / sum(prior * weight)
    """
    #第二种方法
    weight = np.array([assign_weight(p) for p in prior])

    #加载一个损失记录器
    recorder = Loss_recorder('val')

    pruning_rate = 0.3
    # 定义剪枝步骤数和微调步骤数
    num_pruning_steps = 5
    num_finetune_steps = 5

    for epoch in range(1, args.epochs + 1):
        train_prune_quantize(args, model, device, train_loader, val_loader, optimizer, weight, pruning_rate, num_pruning_steps,
                    num_finetune_steps)

    """    
    for epoch in range(1, args.epochs + 1):
        train_net(args, model, device, train_loader, optimizer, epoch, weight)

        loss = test_net(args, model, device, val_loader, weight)
        
        print('epoch finished')
        if args.save_model:
            torch.save(model.state_dict(), "model_shot3.pt")
            print('model is saved at', 'model.pt')
        
        # record loss
        val_rec.take(loss)        
        val_rec.save()
    """