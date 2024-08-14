import os

__all__ = ["proj_root", "arg", "netChannel"]

netChannel = {"VGG": [64, 128, 256, 512, 512],
              "ResNet": [64, 256, 512, 1024, 2048],
              "ResNext": [64, 256, 512, 1024, 2048],
              "inter": [64, 64, 128, 256, 256],
              "final": [64, 64, 64, 64, 64]}

proj_root = os.path.dirname(__file__)  # 返回文件所在的绝对路径，终点是文件所在的文件夹
dataset_root = os.path.join(proj_root, "data/dataset")  # 数据集目录，以下是各个数据集地址

#MOD
GDD_TR = os.path.join(dataset_root, "MOD", "GDD/train")
GDD_VA = os.path.join(dataset_root, "MOD", "GDD/test")
GDD = os.path.join(dataset_root, "MOD", "GDD/test")
SYS = os.path.join(dataset_root, "MOD", "SYS/test")
MOD_Train = os.path.join(dataset_root, "MOD", "MSD/train")
MOD_valid = os.path.join(dataset_root, "MOD", "MSD/test")
Trans10k = os.path.join(dataset_root, "MOD", "Trans10k/test")
MSD = os.path.join(dataset_root, "MOD", "MSD/test")
Trans10k_TR = os.path.join(dataset_root, "MOD", "Trans10k/train")
Trans10k_TE = os.path.join(dataset_root, "MOD", "Trans10k/test")
Trans10k_VA = os.path.join(dataset_root, "MOD", "Trans10k/valid")

COD_TR = os.path.join(dataset_root, "COD", "train")
CAMO = os.path.join(dataset_root, "COD", "test/CAMO")
CHAMELEON = os.path.join(dataset_root, "COD", "test/CHAMELEON")
COD10K = os.path.join(dataset_root, "COD", "test/COD10K")
NC4K = os.path.join(dataset_root, "COD", "test/NC4K")

Polyp_TR = os.path.join(dataset_root, "MOD", "Polyp/train")
CVC300_TE = os.path.join(dataset_root, "MOD", "Polyp/CVC-300")
CVCC_TE = os.path.join(dataset_root, "MOD", "Polyp/CVC-ClinicDB")
CVCCo_TE = os.path.join(dataset_root, "MOD", "Polyp/CVC-ColonDB")
ETIS_TE = os.path.join(dataset_root, "MOD", "Polyp/ETIS-LaribPolypDB")
Kvasir_TE = os.path.join(dataset_root, "MOD", "Polyp/Kvasir")

# RGB
DUTS_TR = os.path.join(dataset_root, "RGB", "DUTS/train")
DUTS_TE = os.path.join(dataset_root, "RGB", "DUTS/test")
DUTO = os.path.join(dataset_root, "RGB", "DUTO")
ECSSD = os.path.join(dataset_root, "RGB", "ECSSD")
PASCAL_S = os.path.join(dataset_root, "RGB", "PASCAL-S")
HKU_IS = os.path.join(dataset_root, "RGB", "HKU-IS")
# RGBD
RGBD_Train = os.path.join(dataset_root, "RGBD", "NJUD+NLPR+DUT-RGBD_TR")
RGBD_valid = os.path.join(dataset_root, "RGBD", "NJUD+NLPR+DUT-RGBD_TE")
NJUD = os.path.join(dataset_root, "RGBD", "NJUD/test")
NLPR = os.path.join(dataset_root, "RGBD", "NLPR/test")
LFSD = os.path.join(dataset_root, "RGBD", "LFSD")
DUT_RGBD = os.path.join(dataset_root, "RGBD", "DUT-RGBD/test")
RGBD135 = os.path.join(dataset_root, "RGBD", "RGBD135")
SIP = os.path.join(dataset_root, "RGBD", "SIP")
SSD = os.path.join(dataset_root, "RGBD", "SSD")
STERE = os.path.join(dataset_root, "RGBD", "STERE")
# STERE797 = os.path.join(dataset_root, "RGBD", "STEREO")


# 配置参数
arg = {
    # load dataset
    "input_size": 384,\

    # 样本输入大小
    "batch_size": 16,  # batch大小
    "num_workers": 4,  # 使用多少子进程加载数据，越大加载越快，但内存开销大。

    # train
    # "backbone": "PVT_v2",  # 主干网路
    "backbone": "ResNet",  # 主干网路

    "lr_strategy": "every_epoch",

    "model_name": "[spanet_model_2]",  # 自己搭建网络模型名字/本次训练保存结果的文件名

    "pretrained": os.path.join(proj_root, "model/res2net50_v1b_26w_4s-3cf99910.pth"),
    # "pretrained": os.path.join(proj_root, "model/resnext101_32x8d-8ba56ff5.pth"),
    #
    # "pretrained": os.path.join(proj_root, "model/res2net50_v1b_26w_4s-3cf99910.pth"),
    "resume": False,
    "epoch": 100,
    "print_fre": 40,  # 这是啥嘛？记录输出间隔的？
    # "optimizer" :"SGD",edge+region
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "nesterov": False,
    "lr_decay": 0.9,
    # test
    "save": True,
    # rgb dataset
    "rgb": {
        "tr_data_path": DUTS_TR,
        "valid_data_path": DUTS_TE,
        "te_data_list": {
            "test": DUTS_TE,
            "DUTO": DUTO,
            "ECSSD": ECSSD,
            "PASCAL-S": PASCAL_S,
            "HKU-IS": HKU_IS,
        },
    },
    "rgbd": {
        "tr_data_path": RGBD_Train,
        "valid_data_path": RGBD_valid,
        "te_data_list": {
            "NJUD": NJUD,
            "NLPR": NLPR,
            "LFSD": LFSD,
            "RGBD135": RGBD135,
            "DUT-RGBD": DUT_RGBD,
            "SIP": SIP,
            "SSD": SSD,
            "STERE1000": STERE,
            # "STERE797": STERE797,
            # "test": RGBD_Train,
            # "valid": RGBD_valid,
        },
    },
    "mod": {
        "tr_data_path": COD_TR,
        "valid_data_path": CHAMELEON,
        "te_data_list": {
        #     "CVC300": CVC300_TE,
        #     "CVCC": CVCC_TE,
        #     "CVCColon": CVCCo_TE,
        #     "ETIS": ETIS_TE,
        #     "Kvasir": Kvasir_TE,



        "CAMO": CAMO,
        "CHAMELEON": CHAMELEON,
        "COD10K": COD10K,
        "NC4K": NC4K,

        # "Trans10k": Trans10k_TE,

        # "MSD": MSD,

        # "tr_data_path": DUTS_TR,
        # "valid_data_path": DUTS_TE,
        # "te_data_list": {
        #     "test": DUTS_TE,
        #     "DUTO": DUTO,
        #     "ECSSD": ECSSD,
        #     "PASCAL-S": PASCAL_S,
        #     "HKU-IS": HKU_IS,

        }
    }


}
