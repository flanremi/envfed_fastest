import json
import os
import random

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import torchvision.models.resnet as resnet

from model_util import get_params_by_url


# 自编码器
class AutoCoder1(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.conv11 = nn.Conv1d(kernel_size=17, stride=1, in_channels=1, out_channels=16, padding=8).to(device)
        self.pool11 = nn.MaxPool1d(stride=16, kernel_size=16, return_indices=True).to(device)
        self.bn11 = nn.BatchNorm1d(16).to(device)
        self.conv12 = nn.Conv1d(kernel_size=17, stride=1, in_channels=16, out_channels=32, padding=8).to(device)
        self.pool12 = nn.MaxPool1d(stride=16, kernel_size=16, return_indices=True).to(device)
        self.bn12 = nn.BatchNorm1d(32).to(device)
        self.conv13 = nn.Conv1d(kernel_size=17, stride=1, in_channels=32, out_channels=64, padding=8).to(device)
        self.pool13 = nn.MaxPool1d(stride=16, kernel_size=16, return_indices=True).to(device)
        self.bn13 = nn.BatchNorm1d(64).to(device)
        self.conv14 = nn.Conv1d(kernel_size=17, stride=1, in_channels=64, out_channels=128, padding=8).to(device)
        self.pool14 = nn.MaxPool1d(stride=16, kernel_size=16, return_indices=True).to(device)
        self.bn14 = nn.BatchNorm1d(128).to(device)

        self.dnn11 = nn.Linear(512, 128).to(device)
        self.dnn12 = nn.Linear(128, 16).to(device)

        self.dnn21 = nn.Linear(16, 128).to(device)
        self.dnn22 = nn.Linear(128, 512).to(device)

        self.conv21 = nn.ConvTranspose1d(kernel_size=17, stride=1, in_channels=128, out_channels=64, padding=8).to(device)
        self.pool21 = nn.MaxUnpool1d(stride=16, kernel_size=16).to(device)
        self.bn21 = nn.BatchNorm1d(64).to(device)
        self.conv22 = nn.ConvTranspose1d(kernel_size=17, stride=1, in_channels=64, out_channels=32, padding=8).to(device)
        self.pool22 = nn.MaxUnpool1d(stride=16, kernel_size=16).to(device)
        self.bn22 = nn.BatchNorm1d(32).to(device)
        self.conv23 = nn.ConvTranspose1d(kernel_size=17, stride=1, in_channels=32, out_channels=16, padding=8).to(device)
        self.pool23 = nn.MaxUnpool1d(stride=16, kernel_size=16).to(device)
        self.bn23 = nn.BatchNorm1d(16).to(device)
        self.conv24 = nn.ConvTranspose1d(kernel_size=17, stride=1, in_channels=16, out_channels=1, padding=8).to(device)
        self.pool24 = nn.MaxUnpool1d(stride=16, kernel_size=16).to(device)
        self.bn24 = nn.BatchNorm1d(1).to(device)

    def forward2(self, x):

        x, indices1 = self.pool11(fun.relu(self.bn11(self.conv11(x))))
        x, indices2 = self.pool12(fun.relu(self.bn12(self.conv12(x))))
        x, indices3 = self.pool13(fun.relu(self.bn13(self.conv13(x))))
        x, indices4 = self.pool14(fun.relu(self.bn14(self.conv14(x))))
        x = x.view(-1, 1, 128 * 4)
        x = fun.relu(self.dnn11(x))
        x = fun.relu(self.dnn12(x))

        x = fun.relu(self.dnn21(x))
        x = fun.relu(self.dnn22(x))

        x = x.view(-1, 128, 4)

        x = fun.relu(self.bn21(self.conv21(self.pool21(x, indices4))))
        x = fun.relu(self.bn22(self.conv22(self.pool22(x, indices3))))
        x = fun.relu(self.bn23(self.conv23(self.pool23(x, indices2))))
        x = fun.relu(self.bn24(self.conv24(self.pool24(x, indices1))))

        return x

    def forward(self, x):
        x, indices1 = self.pool11(fun.relu(self.bn11(self.conv11(x))))
        x, indices2 = self.pool12(fun.relu(self.bn12(self.conv12(x))))
        x, indices3 = self.pool13(fun.relu(self.bn13(self.conv13(x))))
        x, indices4 = self.pool14(fun.relu(self.bn14(self.conv14(x))))
        x = x.view(-1, 1, 128 * 4)
        x = fun.relu(self.dnn11(x))
        x = fun.relu(self.dnn12(x))
        return x
def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames
class AutoCoderModelHelper:

    def __init__(self):
        super().__init__()
        self.env_names = ["crossing", "high_way", "main_road", "total"]
        self.dir_url = "./model_25/{}/client{}/"
        self.pool = []
        self.init_pool()

        self.device = torch.device("cuda:0")
        self.net = AutoCoder1(self.device).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.lossFunc = torch.nn.MSELoss()

    def init_pool(self):
        for env_name in self.env_names:
            for i in range(25):
                dir_url = self.dir_url.format(env_name, i)
                file_names = get_all_filenames(dir_url)
                j = 0
                while j < len(file_names):
                    if file_names[j].find("epoch") == -1:
                        file_names.pop(j)
                    else:
                        j += 1
                self.pool.extend(file_names)


    def ge_squad(self, ):
        tmp = self.pool.copy()
        random.shuffle(tmp)
        _ts = tmp
        result = []
        for _t in _ts:
            item = get_params_by_url(_t)
            result.append(item)
        return result




if __name__ == '__main__':
    helper = AutoCoderModelHelper()
    a = helper.ge_squad()
    result = []
    # 周期
    for i in range(100):

        random.shuffle(a)

        # target = torch.from_numpy(np.array(a[j * 16: (j + 1) * 16])).to(helper.device, torch.float)
        for j in range(int(len(a) / 100)):
            loss = 0
            helper.optimizer.zero_grad()
            b = torch.stack(a[j * 100: (j + 1) * 100], dim=0).to(helper.device, torch.float)
            out = helper.net.forward(b)
            _loss = torch.mean(helper.lossFunc(out, b))
            _loss.backward()
            helper.optimizer.step()
            loss += _loss.to(torch.device("cpu")).detach().numpy().reshape(-1).tolist()[0]
            loss /= 100
            result.append(loss)
            print(loss)
            torch.cuda.empty_cache()
            if j % 10 == 0:
                torch.save(helper.net.state_dict(), "auto_encoder\\model1.pt")
                with open("auto_encoder\\loss.txt", "w+") as f:
                    f.write(json.dumps(result))
