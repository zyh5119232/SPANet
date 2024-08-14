import os
import torch
from datetime import datetime
import numpy as np
from torch import nn
import torch.backends.cudnn as torchcudnn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # 用来显示进度条的模块
import time
from data.createDataset import createLoader
from module.Loss_ssim import SSIM
from tool.misc import AvgLoss, construct_path_dict, make_log
from tool.polypMetric.polypM import CalTotalMetric
from config import proj_root, arg
from module.Loss import IEL
from module.Loss import Polyp

from torch.nn import BCELoss
from net.MyNet_Mirror_3base import My_mirrorNet
from net.mirrornet import MirrorNet
torch.manual_seed(0)  # 为CPU设置种子，生成随机数
torch.cuda.manual_seed_all(0)  # 为所有GPU设置种子，生成随机数
torchcudnn.benchmark = True  # 自动寻找最优算法，实现加速
torchcudnn.enabled = True  # 开启非确定算法 和 benchmark配合使用
torchcudnn.deterministic = True  # 保证每次找的最优算法一致，避免每次实验的结果波动。和固定随机数种子配合使用


# Loss Configureation



def floss(prediction, target, beta=0.3, log_like=False):
    EPS = 1e-10
    prediction = torch.sigmoid(prediction)
    batch_size = prediction.size(0)
    fmatrix = torch.zeros(batch_size, 1).cuda()
    for i in range(batch_size):
        N = N = prediction[i:i + 1, :, :].size(1)
        TP = (prediction[i:i + 1, :, :] * target[i:i + 1, :, :]).view(N, -1).sum(dim=1)
        H = beta * target[i:i + 1, :, :].view(N, -1).sum(dim=1) + prediction[i:i + 1, :, :].view(N, -1).sum(dim=1)
        fmatrix[i] = (1 + beta) * TP / (H + EPS)
    fmeasure = torch.mean(fmatrix).cuda()
    if log_like:
        floss = -torch.log(fmeasure)
    else:
        floss = (1 - fmeasure).cuda()
    return floss


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()
        # config
        self.arg = arg
        # data
        self.tr_loader = createLoader(data_path=self.arg["mod"]["tr_data_path"],
                                      in_size=self.arg["input_size"],
                                      data="train",
                                      mode="RGB") #MOD   RGB
        self.valid_loader = createLoader(data_path=self.arg["mod"]["valid_data_path"],
                                         in_size=self.arg["input_size"],
                                         data="valid",
                                         mode="RGB")
        self.te_data_list = self.arg["mod"]["te_data_list"]  # 数据集名字列表
        self.to_pil = transforms.ToPILImage()  # 将tensor转为PIL，保存后可直接打开图片文件，用于将输出还原为PIL
        # train-setting
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 配置环境
        self.net = My_mirrorNet(base=self.arg["backbone"], path=self.arg["pretrained"]).to(self.dev)  # 搭建网络
        self.loss = Polyp().to(self.dev)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.0001)
        # self.optimizer = self.make_optimizer()
        self.dloss = IEL().to(self.dev)
        self.edgeloss = BCELoss().to(self.dev)
        self.sloss = SSIM(window_size=11, size_average=True)
        # self.loss = MSL().to(self.dev)
        # self.loss = GuideLoss().to(self.dev)
        # self.optimizer = self.make_optimizer()

        # save-path
        self.path = construct_path_dict(proj_root=proj_root, exp_name=self.arg["model_name"])  # 保存输出结果地址
        self.save_path = self.path["pre"]
        self.full_net = os.path.join(self.path["pth"], "full_final.pth.tar")
        self.state_net = os.path.join(self.path["pth"], "state_final.pth")
        self.para_number()  # 获取网络参数并保存
        self.bceloss = nn.BCELoss(reduction='mean')
        # resume-model
        if self.arg["resume"]:  # 是否加载模型
            try:
                self.resume_checkpoint(load_path=self.full_net, mode="all")
            except:
                print(f"{self.full_net} does not exist and we will load {self.state_net}")
                self.resume_checkpoint(load_path=self.state_net, mode="onlynet")


    def train(self):
        best_epoch = 0
        best_loss = 0
        #
        iter_num = len(self.tr_loader)
        make_log(self.path["log_tr"], f"=== log_tr {datetime.now()} ===")  # 打开path路径下的文件，写入当前时间
        for curr_e in range(self.arg["epoch"]):
            if (curr_e == 35):
                break
            if self.arg["lr_strategy"] == "every_epoch":
                self.change_lr(curr_e, self.arg["epoch"])
            else:
                self.adjust_lr(self.arg["lr"], curr_e, decay_epoch=10)
            # train
            self.net.train()  # 设置网络为训练模式
            train_loss_record = AvgLoss()  #??
            for i, data in enumerate(self.tr_loader):
                # pre
                # img, depth, Mask, img_name = data
                # img, mask, body, edge, img_name = data
                # img, teacher, Mask, img_name = data
                img, mask, gt, img_name = data
                img = img.float()
                # depth = depth.float()
                # teacher = teacher.float()
                mask = mask.float()
                gt = gt.float()
                # body = body.float()
                # edge = edge.float()

                img = img.to(self.dev, non_blocking=True)
                # depth = depth.to(self.dev, non_blocking=True)
                # teacher = teacher.to(self.dev, non_blocking=True)
                mask = mask.to(self.dev, non_blocking=True)
                gt = gt.to(self.dev, non_blocking=True)
                # body = body.to(self.dev, non_blocking=True)
                # edge = edge.to(self.dev, non_blocking=True)
                a, label = self.net(img)

                # backward
                self.optimizer.zero_grad()
                # train_loss = self.loss(pre, Mask)

                loss_detail = 10*self.dloss(label, mask)+self.dloss(a, mask)
                # loss_detail = self.loss(detail*gt, mask*gt)*10
                # loss_label = self.loss(label*gt, mask*gt)*10
                train_loss = loss_detail

                torch.cuda.empty_cache()  # 自动回收不用的显存
                train_loss.backward()
                self.optimizer.step()
                # print
                train_iter_loss = train_loss.item()
                train_loss_record.update(train_iter_loss, mask.size(0))  # size(0)得到每个batch图片的张数
                if self.arg["print_fre"] > 0 and (i + 1) % self.arg["print_fre"] == 0:  # 每训练print_fre次进行一次记录
                    log_train = (f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\t"  # 当前时间
                                 f"[{self.arg['model_name']}]>"  # 模型名称
                                 f"[E:{curr_e + 1}/{self.arg['epoch']}][I:{i + 1}/{iter_num}]>"  
                                 # 训练进度(第几个epoch，每个epoch训练到第几个batch)，均用百分数表示
                                 f"[Lr:{self.optimizer.param_groups[0]['lr']:.7f}]"  # 当前学习率
                                 f"[Avg:{train_loss_record.avg:.5f}|Cur:{train_iter_loss:.5f}]")  # 平均损失及当前损失
                    print(log_train)  # 输出
                    make_log(self.path["log_tr"], log_train)  # 记录
            # valid
            self.net.eval()  # 网络改为测试模式，每训练一个epoch就进行验证
            curr_loss = 0
            tqdm_iter = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), leave=False)  # 生成进度条
            for i, data in tqdm_iter:
                tqdm_iter.set_description(f"{self.arg['model_name']}:" f"te=>{i + 1}")  # 配置进度条说明
                with torch.no_grad():
                    # pre
                    img, mask, img_name = data
                    img = img.float()
                    mask = mask.float()
                    img = img.to(self.dev, non_blocking=True)
                    # depth = depth.to(self.dev, non_blocking=True)
                    # teacher = teacher.to(self.dev, non_blocking=True)
                    mask = mask.to(self.dev, non_blocking=True)
                    a, label = self.net(img)
                    pre = F.interpolate(label, size=mask.size()[2:], mode="bilinear", align_corners=True)
                    # 插值，改变预测图大小
                    # Mask = F.interpolate(Mask, size=pre.size()[2:], mode="bilinear", align_corners=True)
                    # loss
                    valid_loss = self.loss(pre, mask)
                    curr_loss += valid_loss.item()
            # save best model
            if curr_e == 0:
                best_loss = curr_loss
            if curr_loss <= best_loss:
                best_loss = curr_loss
                best_epoch = curr_e + 1
                self.save_checkpoint(epoch=best_epoch,
                                     full_net_path=os.path.join(self.path["pth"], "full_best.pth.tar"),
                                     state_net_path=os.path.join(self.path["pth"], "state_best.pth"))
            log_valid = (f"[{self.arg['model_name']}]>"
                         f"[best_epoch: {best_epoch} | curr_epoch: {curr_e + 1}]"
                         f"[best_loss: {best_loss:.5f} | curr_loss: {curr_loss:.5f}]")
            print(log_valid)
            make_log(self.path["log_tr"], log_valid)
            #
            if curr_e >= 4:
                self.net.eval()
                msg = f"\ntesting in the {curr_e+1}th epoch"
                print(msg)
                make_log(self.path["log_te"], msg)
                make_log(self.path["log_te"], f"=== log_te {datetime.now()} ===")

                # miou = 0
                for data_name, data_path in self.te_data_list.items():  # items,将字典的键值对组成元组

                    te_loader = createLoader(data_path=data_path,
                                             in_size=self.arg["input_size"],
                                             data="test",
                                             mode="MOD")

                    save_path = os.path.join(self.save_path, data_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # test
                    cal_total_metrics = CalTotalMetric(num=len(te_loader), beta_for_wfm=1)  # 各个测试指标
                    tqdm_iter = tqdm(enumerate(te_loader), total=len(te_loader), leave=False)  # 进度条
                    for i, data in tqdm_iter:
                        # best_iou = 0
                        tqdm_iter.set_description(f"{self.arg['model_name']}:" f"te=>{i + 1}")
                        with torch.no_grad():
                            # img, depth, mask_path, img_name = data
                            # img, mask_path, img_name = data
                            img, mask, img_name = data
                            img = img.float()
                            # depth = depth.float()
                            img = img.to(self.dev, non_blocking=True)
                            # depth = depth.to(self.dev, non_blocking=True)
                            a, label = self.net(img)
                            pred_array_tensor = label.cpu().detach()  # 将数据转到CPU上并取消正反向传播
                        # cal metric
                        for item, pred in enumerate(pred_array_tensor):
                            # Mask
                            mask_pil = Image.open(mask[item]).convert("L")
                            original_size = mask_pil.size

                            mask_array = np.asarray(mask_pil)
                            mask_array = mask_array / (mask_array.max() + 1e-8)
                            mask_array = np.where(mask_array > 0.5, 1, 0)  # mask_array > 0.5成立返回值为1，否则为0
                            # 可以理解为二值化
                            # pre
                            pre_pil = self.to_pil(pred).resize(original_size, resample=Image.NEAREST)
                            # to_pil，将输出（像素值位于[0,1]）转为PIL（像素值位于[0,255]）
                            pre_array = np.asarray(pre_pil)
                            # Normalized
                            max_pre_array = pre_array.max()
                            min_pre_array = pre_array.min()
                            if max_pre_array == min_pre_array:
                                pre_array = pre_array / 255  # 最大值与最小值相等，即没有显著物体，除以255变为1或0
                            else:
                                pre_array = (pre_array - min_pre_array) / (max_pre_array - min_pre_array)
                            # 这一步也没有进行二值化啊？还是说网络输出的图片只存在0或255
                            cal_total_metrics.update(pre_array, mask_array)
                    # print
                    results = cal_total_metrics.show()
                    pre_results = {k: f"{v:.5f}" for k, v in results.items()}
                    msg = f" ==>> 在{data_name}:'{data_path}'测试集上结果\n >> {pre_results}"
                    print(msg)
                    make_log(self.path["log_te"], msg)
                    self.save_checkpoint(epoch=curr_e + 1,
                                         full_net_path=os.path.join(self.path["pth"], str(curr_e + 1) + ".pth.tar"),
                                         state_net_path=os.path.join(self.path["pth"], str(curr_e + 1) + ".pth"))
                #     miou += results["mIou"]
                # if curr_e == 0:
                #     best_iou = miou
                # if best_iou < miou:
                #     best_iou = miou
                #     print(f"best_iou:{best_iou / 5},epoch:{curr_e}")

            #
        # save final model
        self.save_checkpoint(epoch=self.arg["epoch"],
                             full_net_path=self.full_net,
                             state_net_path=self.state_net)

    def test(self):
        self.net.eval()
        make_log(self.path["log_te"], f"=== log_te {datetime.now()} ===")
        for data_name, data_path in self.te_data_list.items():  # items,将字典的键值对组成元组

            te_loader = createLoader(data_path=data_path,
                                     in_size=self.arg["input_size"],
                                     data="test",
                                     mode="RGB")

            save_path = os.path.join(self.save_path, data_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # test
            cal_total_metrics = CalTotalMetric(num=len(te_loader), beta_for_wfm=1)  # 各个测试指标
            tqdm_iter = tqdm(enumerate(te_loader), total=len(te_loader), leave=False)  # 进度条
            for i, data in tqdm_iter:
                tqdm_iter.set_description(f"{self.arg['model_name']}:" f"te=>{i + 1}")
                with torch.no_grad():
                    # img, depth, mask_path, img_name = data
                    img, mask_path, img_name = data
                    img = img.float()
                    # depth = depth.float()
                    img = img.to(self.dev, non_blocking=True)
                    # depth = depth.to(self.dev, non_blocking=True)
                    # pre = self.net(img,depth)
                    detail, label = self.net(img)
                pred_array_tensor = label.cpu().detach()  # 将数据转到CPU上并取消正反向传播
                # cal metric
                for item, pred in enumerate(pred_array_tensor):
                    # Mask
                    mask_pil = Image.open(mask_path[item]).convert("L")
                    original_size = mask_pil.size
                    mask_array = np.asarray(mask_pil)
                    mask_array = mask_array / (mask_array.max() + 1e-8)
                    mask_array = np.where(mask_array > 0.5, 1, 0)  # mask_array > 0.5成立返回值为1，否则为0
                    # 可以理解为二值化
                    # pre
                    pre_pil = self.to_pil(pred).resize(original_size, resample=Image.NEAREST)
                    # to_pil，将输出（像素值位于[0,1]）转为PIL（像素值位于[0,255]）
                    pre_array = np.asarray(pre_pil)
                    # Normalized
                    max_pre_array = pre_array.max()
                    min_pre_array = pre_array.min()
                    if max_pre_array == min_pre_array:
                        pre_array = pre_array / 255  # 最大值与最小值相等，即没有显著物体，除以255变为1或0
                    else:
                        pre_array = (pre_array - min_pre_array) / (max_pre_array - min_pre_array)
                        # 这一步也没有进行二值化啊？还是说网络输出的图片只存在0或255
                    cal_total_metrics.update(pre_array, mask_array)
                    # save
                    if self.arg["save"]:
                        pre_path = os.path.join(save_path, img_name[item] + ".png")
                        pre_pil.save(pre_path)
            # print
            results = cal_total_metrics.show()
            pre_results = {k: f"{v:.3f}" for k, v in results.items()}
            msg = f" ==>> 在{data_name}:'{data_path}'测试集上结果\n >> {pre_results}"
            print(msg)
            make_log(self.path["log_te"], msg)

    def change_lr(self, curr, total_num):
        ratio = pow((1 - float(curr) / total_num), self.arg["lr_decay"])

        self.optimizer.param_groups[0]["lr"] = self.optimizer.param_groups[0]["lr"] * ratio

    def adjust_lr(self, init_lr, epoch, decay_rate=0.1, decay_epoch=10):
        decay = decay_rate ** (epoch // decay_epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay

    def make_optimizer(self):
        # https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_tricks.py
        params = [
            {"params": [p for name, p in self.net.named_parameters() if ("bias" in name or "bn" in name)],
             "weight_decay": 0},
            {"params": [p for name, p in self.net.named_parameters() if ("bias" not in name and "bn" not in name)]}
        ]
        print(self.net.named_parameters())
        if (self.arg["optimizer"] == "SGD"):
            optimizer = torch.optim.SGD(params,
                                        lr=self.arg["lr"],
                                        momentum=self.arg["momentum"],
                                        weight_decay=self.arg["weight_decay"],
                                        nesterov=self.arg["nesterov"])
            return optimizer
        if (self.arg["optimizer"] == "AdamW"):
            optimizer = torch.optim.AdamW(params=self.net.parameters(), lr=self.arg["lr"])
            return optimizer


    def save_checkpoint(self, epoch, full_net_path, state_net_path):
        """保存模型"""
        state_dict = {"epoch": epoch,
                      "net_state": self.net.state_dict()}
        torch.save(state_dict, state_net_path)
        state_dict["opti_state"] = self.optimizer.state_dict()  # 优化器字典，保存当前状态，可用于恢复训练
        torch.save(state_dict, full_net_path)
        # 保存完整模型就比保存数据多了优化器

    def resume_checkpoint(self, load_path, mode="all"):
        if os.path.exists(load_path) and os.path.isfile(load_path):  # isfile，判断所给路径是否为文件
            checkpoint = torch.load(load_path, map_location=self.dev)  # 加载模型到self.dev设备上
            epoch = checkpoint["epoch"]
            if mode == "all":
                self.net.load_state_dict(checkpoint["net_state"])
                self.optimizer.load_state_dict(checkpoint["opti_state"])
                print(f" ==> loaded checkpoint '{load_path}' (epoch: {epoch})<<== ")
            elif mode == "onlynet":
                self.net.load_state_dict(checkpoint["net_state"])
                print(f"only has the net's weight params\n"
                      f" ==> loaded checkpoint '{load_path}' (epoch: {epoch}) <<== ")
            else:
                raise NotImplementedError
        else:
            raise Exception(f"{load_path}路径不正常，请检查")

    def para_number(self):
        """获取网络参数数量，并写入到log_te文件中"""
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        print_num = f"The number of parameters: {num_params}"
        print(print_num)
        make_log(self.path["log_te"], print_num)


if __name__ == "__main__":
    model = Trainer()

    print(f" ===========>> {datetime.now()}: 开始训练 <<=========== ")
    model.train()
    print(f" ===========>> {datetime.now()}: 结束训练 <<=========== ")
    #
    # print(f" ===========>> {datetime.now()}: 开始测试 <<=========== ")
    # model.test()
    # print(f" ===========>> {datetime.now()}: 结束测试 <<=========== ")

