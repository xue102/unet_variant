import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2
from model.unet_model import UNet
from model.MSA_unet import MSA_UNet
from model.DC_ResUNet import DC_ResUNet


def Dice(pred, target):
    pred = F.sigmoid(pred)
    # Dice
    smooth = 1e-6
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def main(args):
    # 加载模型
    if args.model == "unet":
        net = UNet(n_channels=1, n_classes=1).to(args.device)
    elif args.model == "msa_unet":
        net = MSA_UNet(in_channels=1, out_channels=1).to(args.device)
    else:
        net = DC_ResUNet(in_channels=1, out_channels=1).to(args.device)

    # 加载模型参数
    net.load_state_dict(torch.load(f'{args.model}_best_model.pth', map_location=args.device))
    # 测试模式
    net.eval()
    # 读取图片
    img = cv2.imread(args.data_path)
    tar = cv2.imread(args.data_path.replace("image", "label"))
    # 转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tar = cv2.cvtColor(tar, cv2.COLOR_RGB2GRAY)
    # 转为batch为1，通道为1，大小为512*512的数组
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    tar = tar.reshape(1, 1, tar.shape[0], tar.shape[1])
    # 处理标签，将像素值为255的改为1
    if tar.max() > 1:
        tar = tar / 255
    # 转为tensor
    img_tensor = torch.from_numpy(img)
    tar_tensor = torch.from_numpy(tar)
    # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    img_tensor = img_tensor.to(device=args.device, dtype=torch.float32)
    tar_tensor = tar_tensor.to(device=args.device, dtype=torch.float32)
    # 预测
    pred = net(img_tensor)
    dice = Dice(pred, tar_tensor).item()
    print(f"{args.model}: Dice {dice}")
    # 提取结果
    pred = np.array(pred.data.cpu()[0])[0]
    # # 处理结果
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    # 保存图片
    cv2.imwrite(f"{args.data_path.split('/')[-1][:-4]}_{args.model}_res.png", pred)


if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='Cell Segmentation Predicting')
    parser.add_argument('--model', type=str, choices=['unet', 'dc_resunet'], default='dc_resunet',
                        help='Model for predicting')
    parser.add_argument('--data_path', type=str, default='./data/train/image/10.png', help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # 打印参数配置
    print("\nPredicting Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n")

    main(args)

    # # 选择设备，有cuda用cuda，没有就用cpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # 加载网络，图片单通道，分类为1。
    # net = UNet(n_channels=1, n_classes=1)
    # # 将网络拷贝到deivce中
    # net.to(device=device)
    # # 加载模型参数
    # net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # # 测试模式
    # net.eval()
    # # 读取所有图片路径
    # tests_path = glob.glob('data/test/*.png')
    # # 遍历素有图片
    # for test_path in tests_path:
    #     # 保存结果地址
    #     save_res_path = test_path.split('.')[0] + '_res.png'
    #     # 读取图片
    #     img = cv2.imread(test_path)
    #     # 转为灰度图
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     # 转为batch为1，通道为1，大小为512*512的数组
    #     img = img.reshape(1, 1, img.shape[0], img.shape[1])
    #     # 转为tensor
    #     img_tensor = torch.from_numpy(img)
    #     # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    #     img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    #     # 预测
    #     pred = net(img_tensor)
    #     # 提取结果
    #     pred = np.array(pred.data.cpu()[0])[0]
    #     # 处理结果
    #     pred[pred >= 0.5] = 255
    #     pred[pred < 0.5] = 0
    #     # 保存图片
    #     cv2.imwrite(save_res_path, pred)

