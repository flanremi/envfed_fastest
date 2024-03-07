import json
import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
from numpy.testing._private.utils import print_assert_equal

import torch
from torch import optim
from torch.utils.data import dataset
from numpy.core.fromnumeric import shape

from torchsummary import summary

import utils.loss
import utils.utils
import utils.datasets
import model.detector as detector


class Client:

    def __init__(self, data_url, name):
        super().__init__()
        self.name = name
        self.gather_times = 0
        self.val_result = []
        self.cfg = utils.utils.load_datafile(data_url)
        self.dir_url = self.cfg["model_name"]
        if not os.path.exists(self.dir_url):
            os.makedirs(self.dir_url)
            # 数据集加载
        self.train_dataset = utils.datasets.TensorDataset(self.cfg["train"], self.cfg["width"], self.cfg["height"], imgaug=True)
        self.val_dataset = utils.datasets.TensorDataset(self.cfg["val"], self.cfg["width"], self.cfg["height"], imgaug=False)
        self.batch_size = int(self.cfg["batch_size"] / self.cfg["subdivisions"])
        self.nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])
        # 训练集
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       collate_fn=utils.datasets.collate_fn,
                                                       num_workers=self.nw,
                                                       pin_memory=True,
                                                       drop_last=True,
                                                       persistent_workers=True
                                                       )
        # 验证集
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=False,
                                                     collate_fn=utils.datasets.collate_fn,
                                                     num_workers=self.nw,
                                                     pin_memory=True,
                                                     drop_last=False,
                                                     persistent_workers=True
                                                     )

        # 指定后端设备CUDA&CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化模型结构
        self.model = detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)
        summary(self.model, input_size=(3, self.cfg["height"], self.cfg["width"]))
        premodel_path = self.cfg["pre_weights"]
        self.model.load_state_dict(torch.load(premodel_path, map_location=self.device), strict=False)

        # 构建SGD优化器
        self.optimizer = optim.SGD(params=self.model.parameters(),
                              lr=self.cfg["learning_rate"],
                              momentum=0.949,
                              weight_decay=0.0005,
                              )

        # 学习率衰减策略
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                   milestones=self.cfg["steps"],
                                                   gamma=0.1)


    def train(self):
        batch_num = 0
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            pbar = tqdm(self.train_dataloader)

            for imgs, targets in pbar:
                # 数据预处理
                imgs = imgs.to(self.device).float() / 255.0
                targets = targets.to(self.device)

                # 模型推理
                preds = self.model(imgs)
                # loss计算
                iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(preds, targets, self.cfg, self.device)

                # 反向传播求解梯度
                total_loss.backward()

                # 学习率预热
                for g in self.optimizer.param_groups:
                    warmup_num = 5 * len(self.train_dataloader)
                    if batch_num <= warmup_num:
                        scale = math.pow(batch_num / warmup_num, 4)
                        g['lr'] = self.cfg["learning_rate"] * scale

                    lr = g["lr"]

                # 更新模型参数
                if batch_num % self.cfg["subdivisions"] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # 打印相关信息
                info = "Epoch:%d LR:%f CIou:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, lr, iou_loss, obj_loss, cls_loss, total_loss)
                pbar.set_description(info)

                batch_num += 1


            # # 学习率调整
            # self.scheduler.step()
        precision, recall, AP, f1 = self.val()
        self.val_result.append((precision, recall, AP, f1))
        torch.save(self.model.state_dict(), self.cfg["model_name"] + "gather_{}.pth".format(self.gather_times))
        self.gather_times += 1

    def val(self):
        self.model.eval()
        # 模型评估
        print("computer mAP...")
        _, _, AP, _ = utils.utils.evaluation(self.val_dataloader, self.cfg, self.model, self.device)
        print("computer PR...")
        precision, recall, _, f1 = utils.utils.evaluation(self.val_dataloader, self.cfg, self.model, self.device, 0.3)
        print("Precision:%f Recall:%f AP:%f F1:%f" % (precision, recall, AP, f1))
        return precision, recall, AP, f1



