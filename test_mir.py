import os
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from net.TestNet import My_mirrorNet
from data.createDataset import createLoader
from tool.misc import construct_path_dict, make_log
from tool.polypMetric.polypM import CalTotalMetric
from config import proj_root, arg
# # 使用可视化
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# # 使用可视化   tensorboard --logdir=Log/
# from module.visual import draw_features

class Tester:
    def __init__(self):
        super(Tester, self).__init__()
        # config
        self.arg = arg
        self.te_data_list = self.arg["mod"]["te_data_list"]  # 测试集列表
        self.to_pil = transforms.ToPILImage()
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # net
        self.net = My_mirrorNet(base=self.arg["backbone"], path=self.arg["pretrained"]).to(self.dev)
        # save-path
        self.path = construct_path_dict(proj_root=proj_root, exp_name=self.arg["model_name"])
        self.save_path = self.path["pre"]
        self.full_net = os.path.join(self.path["pth"], "34.pth.tar")
        self.state_net = os.path.join(self.path["pth"], "34.pth")
        self.para_number()
        # resume-model
        try:
            self.resume_checkpoint(load_path=self.full_net, mode="all")
            print('加载模型成功')
        except:
            print(f"{self.full_net} does not exist and we will load {self.state_net}")
            self.resume_checkpoint(load_path=self.state_net, mode="onlynet")

    def test(self):

        self.net.eval()
        # msg = f"\ntesting in the {curr_e + 1}th epoch"
        # print(msg)
        # make_log(self.path["log_te"], msg)
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
                    # save
                    if self.arg["save"]:
                        pre_path = os.path.join(save_path, img_name[item] + ".png")
                        pre_pil.save(pre_path)
            # print
            results = cal_total_metrics.show()
            pre_results = {k: f"{v:.5f}" for k, v in results.items()}
            msg = f" ==>> 在{data_name}:'{data_path}'测试集上结果\n >> {pre_results}"
            print(msg)
            make_log(self.path["log_te"], msg)

                    # for item1 in range(4):
                    #     feature_e = feature[0][item1][item].cpu().detach()
                    #     bn = feature[1][item1].cpu().detach()
                    #     featureName = os.path.join(save_path, img_name[item] + f"e{item1+1}.png")
                    #     draw_features(feature_e, featureName, bn, mask_path[item])
            # print
            # 可视化
            #writer.add_graph(self.net, img)
            # writer.add_image('train/pre', pre[0].cpu().detach())
            # writer.add_image('train/img', img[0].cpu().detach())



            # 可视化
            # results = cal_total_metrics.show()
            # pre_results = {k: f"{v:.4f}" for k, v in results.items()}
            # msg = f" ==>> 在{data_name}:'{data_path}'测试集上结果\n >> {pre_results}"
            # print(msg)
            # make_log(self.path["log_te"], msg)

    def resume_checkpoint(self, load_path, mode="all"):
        if os.path.exists(load_path) and os.path.isfile(load_path):
            checkpoint = torch.load(load_path, map_location=self.dev)
            epoch = checkpoint["epoch"]
            if mode == "all":
                #self.net.load_state_dict(checkpoint['state_dict'], False)
                self.net.load_state_dict(checkpoint["net_state"])
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
        num_params = 0
        for p in self.net.parameters():
            num_params += p.numel()
        print_num = f"The number of parameters: {num_params}"
        print(print_num)
        make_log(self.path["log_te"], print_num)


if __name__ == "__main__":
    model = Tester()
    # writer = SummaryWriter("./Log")
    print(f" ===========>> {datetime.now()}: 开始测试 <<=========== ")
    model.test()
    print(f" ===========>> {datetime.now()}: 结束测试 <<=========== ")
