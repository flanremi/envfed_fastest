import torch
import torch.nn.functional as F

# 把权重转化为502*502=252004的二维图像
def get_params_by_url(url):
    model = torch.load(url)
    tmp = []
    for key, value in model.items():
        tmp.append(value.view(-1))
    tmp = torch.cat(tmp)
    tmp = F.pad(tmp, (0, 262144-251664), "constant", 0)
    tmp = tmp.reshape([1,1,262144])
    return tmp

