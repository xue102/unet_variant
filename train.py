import argparse
import os
from torch.utils.data import SubsetRandomSampler
from model.unet_model import UNet
from model.MSA_unet import MSA_UNet
from model.DC_ResUNet import DC_ResUNet
from utils.dataset import ISBI_Loader
from utils.loss import CombinedLogitsLoss
from torch import optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold


def Dice(pred, target):
    pred = F.sigmoid(pred)
    # Dice
    smooth = 1e-6
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def save_metric(save_path, fold, data_iter, args):
    if os.path.exists(save_path):
        with open(save_path, "a", encoding="utf-8") as f:
            f.write("\n" + "-----------------------------------" + "\n")
            f.write(f"{fold}折" + "\n")
            f.write("\n".join(map(str, data_iter)))
    else:
        os.makedirs(f"./logs/{args.model}", exist_ok=True)
        os.mknod(save_path)
        with open(save_path, "w", encoding="utf-8") as f:
            args_list = [f"{arg}:{getattr(args, arg)}" for arg in vars(args)]
            f.write("\n".join(args_list))
            f.write("\n" + "-----------------------------------" + "\n")
            f.write(f"{fold}折" + "\n")
            f.write("\n".join(map(str, data_iter)))


def train_net(train_loader, val_loader, args, fold, best_loss):
    # 加载模型
    if args.model == "unet":
        net = UNet(n_channels=1, n_classes=1).to(args.device)
    elif args.model == "msa_unet":
        net = MSA_UNet(in_channels=1, out_channels=1).to(args.device)
    else:
        net = DC_ResUNet(in_channels=1, out_channels=1).to(args.device)

    if fold == 0:
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"可训练参数量: {trainable_params:,}")

    # 定义优化算法
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

    # 定义Loss算法
    if args.loss == "BCEloss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = CombinedLogitsLoss()

    # 每轮epoch的loss统计和dice统计
    loss_iter = []
    dice_iter = []

    # 训练epochs次
    for epoch in range(args.epochs):
        # sum_loss统计
        sum_loss = 0
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=args.device, dtype=torch.float32)
            label = label.to(device=args.device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            sum_loss += loss.item()
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), f'{args.model}_best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        loss_iter.append(sum_loss / (len(train_loader) * args.batch_size))

        # sum_dice统计
        sum_dice = 0
        net.eval()
        for image, label in val_loader:
            # 将数据拷贝到device中
            image = image.to(device=args.device, dtype=torch.float32)
            label = label.to(device=args.device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算dice
            dice = Dice(pred, label).item()
            sum_dice += dice
        dice_iter.append(sum_dice / (len(val_loader) * args.batch_size))

        print(f"{fold}折-{epoch}/{args.epochs}: loss {loss_iter[-1]}；dice {dice_iter[-1]}")

    save_path = f"./logs/{args.model}/Metric_loss.txt"
    save_metric(save_path, fold, loss_iter, args)
    save_path = f"./logs/{args.model}/Metric_dice.txt"
    save_metric(save_path, fold, dice_iter, args)

    return best_loss


def k_fold(args):
    # 加载训练集
    isbi_dataset = ISBI_Loader(args.data_dir)

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)  # k折，打乱数据
    for fold, (train_idx, val_idx) in enumerate(kf.split(isbi_dataset)):
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                                   batch_size=args.batch_size,
                                                   sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                                 batch_size=args.batch_size,
                                                 sampler=val_subsampler)
        best_loss = train_net(train_loader, val_loader, args, fold, best_loss)


if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='Cell Segmentation Training')
    parser.add_argument('--model', type=str, choices=['unet', 'msa_unet', 'dc_resunet'], default='dc_resunet', help='Model for training')
    parser.add_argument('--loss', type=str, choices=['BCEloss', 'Combinedloss'], default='BCEloss',
                        help='Loss for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--k', type=int, default=5, help='Number of k fold')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./data/train', help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # 打印参数配置
    print("\nTraining Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n")

    k_fold(args)
