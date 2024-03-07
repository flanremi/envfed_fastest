# getPCA25()
import json
import time

from enum import Enum

import numpy as np
import torch
import torch.nn as nn


import os
import torch
import argparse
from tqdm import tqdm

from torchsummary import summary
from pca_helper import get_pca_by_model
from model_util import get_params_by_model, sum_model

import utils.utils
import utils.datasets
import model.detector as detector

class Type(Enum):
    crossing = "crossing"
    high_way = "high_way"
    main_road = "main_road"
    total = "total"
class Environment:

    def __init__(self, _type:str):
        super().__init__()
        # self.meta_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\metadata"
        self.device = torch.device("cuda:0")
        self.latency = [0.4987, 0.5822, 0.4846, 0.4974, 0.3972, 0.2352, 0.4582, 0.1177, 0.5522, 0.2663,
                        0.5906, 0.5494, 0.1386, 0.3238, 0.5497, 0.4256, 0.5062, 0.2681, 0.1058, 0.3328,
                        0.1, 0.5825, 0.2939, 0.3625, 0.2782]
        self.client_num = 25
        self._type = _type
        self.lamda = 0.5
        epochs = [16,32,64,128,256]
        models_url = "./model_25/high_way/client{}/epoch_{}.pth"
        self.models = [torch.load(models_url.format(i, epochs[i % 5] - 1), map_location=self.device) for i in range(25)]
        self.now_model = None
        self.end = 5
        self.val_helper = ValHelper()

        # 标记client是否被选中
        self.tag = [0 for i in range(self.client_num)]
        self.now_ap50 = 0
        self.last_reward = 0
        self.step = 0
        self.selected = 0
        self.punishment = 0


    def reset(self):
        self.step = 0
        self.now_model = None
        self.tag = [0 for i in range(self.client_num)]
        self.last_reward = 0
        self.selected = 0
        self.punishment = 0


    def get_latency(self):
        tmp = 0
        for i in range(self.client_num):
            if self.tag[i] == 1 and self.latency[i] > tmp:
                tmp = self.latency[i]
        return tmp

    def get_state(self):
        states = []
        # 添加當前模型的向量值
        if self.now_model is None:
            states.extend([0 for i in range(10)])
            pca25 = get_pca_by_model([get_params_by_model(model).cpu().numpy() for model in self.models])
            # 添加模型的向量值
            for i in range(self.client_num):
                if self.tag[i] == 1:
                    states.extend([0 for i in range(10)])
                    states.append(0)
                else:
                    states.extend(pca25[i])
                    states.append(self.latency[i])

        else:
            # todo 展开模型
            models = [get_params_by_model(self.now_model).cpu().numpy()]
            models.extend([get_params_by_model(model).cpu().numpy() for model in self.models])
            # models now + 25 = 26
            pca26 = get_pca_by_model(models)
            states.extend(pca26[0])
            # 添加模型的向量值
            for i in range(self.client_num):
                if self.tag[i] == 1:
                    states.extend([0 for i in range(10)])
                    states.append(0)
                else:
                    states.extend(pca26[i + 1])
                    states.append(self.latency[i])

        return  states
    # 10 +  25 * (10+1)

    def get_reward(self):
        self.now_ap50 = self.val_helper.val(self.now_model)[2]
        return self.lamda * self.now_ap50 - (1 - self.lamda) * self.get_latency()

    # 目前定义25个维度 action是從0開始的
    def next(self, action):
        valid = True
        if self.tag[action] == 0:
            # 融合
            if self.step == 0:
                self.now_model = self.models[action]
            else:
                # todo 叠加模型
                self.now_model = sum_model(self.now_model,self.models[action],self.step)
            self.step += 1
            self.tag[action] = 1
        else:
            valid = False

        if self.step == self.end:
            self.last_reward = self.get_reward()
            return self.get_state(), self.last_reward, 1, False
        return self.get_state(), 0.1 if valid else -0.1, 0, valid

class ValHelper:
    def __init__(self):
        self.data_url = "./dqn.data"
        self.cfg = utils.utils.load_datafile(self.data_url)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = detector.Detector(self.cfg["classes"], self.cfg["anchor_num"], True).to(self.device)

        val_dataset = utils.datasets.TensorDataset(self.cfg["val"], self.cfg["width"], self.cfg["height"], imgaug=False)
        batch_size = int(self.cfg["batch_size"] / self.cfg["subdivisions"])
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     collate_fn=utils.datasets.collate_fn,
                                                     num_workers=nw,
                                                     pin_memory=True,
                                                     drop_last=False,
                                                     persistent_workers=True
                                                     )


    def val(self, state_dict):
        self.model.load_state_dict(state_dict)
        result = [0,0,0,0]
        self.model.eval()
        # 模型评估
        _, _, AP, _ = utils.utils.evaluation(self.val_dataloader, self.cfg, self.model, self.device)
        precision, recall, _, f1 = utils.utils.evaluation(self.val_dataloader, self.cfg, self.model, self.device, 0.3)
        result[0] = precision
        result[1] = recall
        result[2] = AP
        result[3] = f1
        return result


if __name__ == '__main__':
    area = Type.high_way.value
    # sigma = 0.3  # 高斯噪声标准差
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    total_step = 0
    env = Environment(area)
    env.lamda = 0.5
    env.now_model = env.models[3]
    print(env.val_helper.val(env.now_model))
    env.now_model = sum_model(env.now_model,env.models[9],0)
    print(env.val_helper.val(env.now_model))
    env.now_model = sum_model(env.now_model,env.models[14],1)
    print(env.val_helper.val(env.now_model))
    env.now_model = sum_model(env.now_model,env.models[19],2)
    print(env.val_helper.val(env.now_model))
    env.now_model = sum_model(env.now_model,env.models[24],3)
    print(env.val_helper.val(env.now_model))
