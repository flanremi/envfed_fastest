import json
import random

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series
import numpy as np
plt.rcParams['pdf.fonttype'] = 42
text_size = 19
fig, ax = plt.subplots()
colors = ['#2878b5', '#9ac9db', '#7fa5b7', '#1ebc61', '#f26647', "#AD11F5"]

# fig1-2， 迭代loss和reward(模型的reward)图， 线图
# 4根线： 3个场景和average

# fig3-4，mdp效果图， 同样是迭代loss和reward, 线图
# 6根线：3个场景,每个场景2根
# 100个epoch，全部25个client模型融合loss和reward,对比训练的fl的loss和reward

# fig5-6， kde图，3种场景下+average的图像准确率和图像识别数，横坐标看情况来，暂定可以离散为50-59,60-69,70-79,80-89,90-100和1,2,3,4,5,>5


# fig7-8,λ效果图,分为0,0.25,0.5,0.75,1五档，箱型线图，输入为5种档位下选择的节点聚合后的图像准确率（只取yolo的百分比即可，无需真实）和图像识别数
# 该图先训练出5个档位的mdp，然后随便选3个场景中的一个来得出点即可

#metadata 0 prec 4 loss
def test():
    with open("./dqn/test", "r") as file:
        data = json.loads(file.read())
    ax.plot([i for i in range(1, 5001)], data, linestyle='-', )
    plt.show()
test()


def loss_prec():
    with open("model_val_data", "r")as file:
        data = json.loads(file.read())
    road_type = ["crossing", "high_way", "main_road", "total"]
    precs = []
    for pos, road_typ in enumerate(road_type):
        _data = data[pos]
        prec = []
        for i in range(256):
            _prec = 0
            for j in range(25):
                client = _data[j]
                _prec += client[i][2]
            prec.append(_prec / 25)
        precs.append(prec)

    x = [i for i in range(1, 257)]

    # 第一组折线，使用左侧Y轴
    ax.plot(x, precs[0],linestyle='-', label='precision_' + road_type[0])  # 绿色实线
    ax.plot(x, precs[1],linestyle='-', label='precision_' +road_type[1])  # 绿色实线
    ax.plot(x, precs[2],linestyle='-', label='precision_' +road_type[2])  # 绿色实线
    ax.plot(x, precs[3],linestyle='-', label='precision_' +road_type[3])  # 绿色实线
    ax.set_xlabel('Period')
    ax.set_ylabel('Precision')
    # ax.tick_params(axis='y', labelcolor='g')


    # 可选：增加图例
    fig.legend(loc='center right', bbox_to_anchor=(0.9,0.4))

    plt.show()


def loss_prec2():
    with open("model_val_data", "r")as file:
        data = json.loads(file.read())
    road_type = ["crossing", "high_way", "main_road", "total"]
    period = [16, 32, 64, 128, 256]
    precs = []
    for pos, road_typ in enumerate(road_type):
        _data = data[pos]
        prec = []
        for i in range(256):
            _prec = 0
            size = 0
            for j in range(25):
                client = _data[j]
                if i < period[j % 5]:
                    _prec += client[i][2]
                    size += 1
            prec.append(_prec / size)
        precs.append(prec)

    x = [i for i in range(1, 257)]

    # 第一组折线，使用左侧Y轴
    ax.plot(x, precs[0],linestyle='-', label='precision_' + road_type[0])  # 绿色实线
    ax.plot(x, precs[1],linestyle='-', label='precision_' +road_type[1])  # 绿色实线
    ax.plot(x, precs[2],linestyle='-', label='precision_' +road_type[2])  # 绿色实线
    ax.plot(x, precs[3],linestyle='-', label='precision_' +road_type[3])  # 绿色实线
    ax.set_xlabel('Period')
    ax.set_ylabel('AP50')
    # ax.tick_params(axis='y', labelcolor='g')


    # 可选：增加图例
    fig.legend(loc='center right', bbox_to_anchor=(0.9,0.4))

    plt.show()

def draw_dqn_reward_loss(pos):
    # pos = 2
    rate = ["0.1", "0.3", "0.5", "0.7","0.9",]
    model_type = ["dqn", "1-step", "3-step", "5-step"]
    urls = ["dqn5/crossing_{}_result_5", "dqn5/crossing_{}_1step_result_5", "dqn5/crossing_{}_3step_result_5",
           "dqn5/crossing_{}_5step_result_5",]
    datas = []
    for url in urls:
        with open(url.format(rate[pos]), "r") as file:
            data = json.loads(file.read())
            datas.append((data.get("reward"), data.get("loss")))


    # rewards = [[datas[i][0][j * 5 + pos] if datas[i][0][j * 5 + pos] != -100 else np.nan
    #             for j in range(int(len(datas[i][0]) / 5))] for i in range(len(datas))]
    # losses = [[datas[i][1][j * 5 + pos] if datas[i][1][j * 5 + pos] != 1 else np.nan
    #            for j in range(int(len(datas[i][1]) / 5))] for i in range(len(datas))]

    rewards = [[datas[i][0][j] if datas[i][0][j] != -100 else np.nan
                for j in range(int(len(datas[i][0])))] for i in range(len(datas))]
    losses = [[datas[i][1][j] if datas[i][1][j] != 1 else np.nan
               for j in range(int(len(datas[i][1])))] for i in range(len(datas))]

    x = [i for i in range(1, 501)]

    # 第一组折线，使用左侧Y轴
    ax.plot(x, rewards[0], linestyle='-', label='reward_' + model_type[0])  # 绿色实线
    ax.plot(x, rewards[1], linestyle='-', label='reward_' + model_type[1])  # 绿色实线
    ax.plot(x, rewards[2], linestyle='-', label='reward_' + model_type[2])  # 绿色实线
    ax.plot(x, rewards[3], linestyle='-', label='reward_' + model_type[3])  # 绿色实线
    ax.set_xlabel('Period')
    # ax.set_ylim(-5,5)
    ax.set_ylabel('Rewards')
    ax.tick_params(axis='y', labelcolor='g')

    # 创建共享X轴的第二个Y轴
    # ax2 = ax.twinx()
    # ax2.plot(x, losses[0], linestyle='-.', label='loss_' + model_type[0])  # 蓝色实线
    # ax2.plot(x, losses[1], linestyle='-.', label='loss_' + model_type[1])  # 蓝色实线
    # ax2.plot(x, losses[2], linestyle='-.', label='loss_' + model_type[2])  # 蓝色实线
    # ax2.plot(x, losses[3], linestyle='-.', label='loss_' + model_type[3])  # 蓝色实线
    # ax2.set_ylabel('Loss')
    # ax2.set_ylim(0,0.4)

    # ax2.tick_params(axis='y', labelcolor='b')

    # 可选：增加图例
    fig.legend()
    plt.title(rate[pos])
    plt.show()


def draw_dqn_step_latency(pos):
    # pos = 1
    rate = ["0.1", "0.3", "0.5", "0.7","0.9",]
    model_type = ["dqn", "1-step", "3-step", "5-step"]
    urls = ["dqn4/crossing_result_3", "dqn4/crossing_1step_result_3", "dqn4/crossing_3step_result_3",
           "dqn4/crossing_5step_result_3",]
    datas = []
    for url in urls:
        with open(url, "r") as file:
            data = json.loads(file.read())
            datas.append((data.get("steps"), data.get("latency")))


    steps = [[datas[i][0][j * 5 + pos] for j in range(int(len(datas[i][0]) / 5))] for i in range(len(datas))]
    latency = [[datas[i][1][j * 5 + pos] for j in range(int(len(datas[i][1]) / 5))] for i in range(len(datas))]

    x = [i for i in range(1, 101)]
    ax.clear()
    # 第一组折线，使用左侧Y轴
    # ax.plot(x, steps[0], linestyle='-', label='steps_' + model_type[0])  # 绿色实线
    # ax.plot(x, steps[1], linestyle='-', label='steps_' + model_type[1])  # 绿色实线
    # ax.plot(x, steps[2], linestyle='-', label='steps_' + model_type[2])  # 绿色实线
    # ax.plot(x, steps[3], linestyle='-', label='steps_' + model_type[3])  # 绿色实线
    # ax.set_xlabel('Period')
    # # ax.set_ylim(-5,)
    # ax.set_ylabel('Steps')
    # ax.tick_params(axis='y', labelcolor='g')

    # 创建共享X轴的第二个Y轴
    ax.plot(x, latency[0], linestyle='-.', label='latency_' + model_type[0])  # 蓝色实线
    ax.plot(x, latency[1], linestyle='-.', label='latency_' + model_type[1])  # 蓝色实线
    ax.plot(x, latency[2], linestyle='-.', label='latency_' + model_type[2])  # 蓝色实线
    ax.plot(x, latency[3], linestyle='-.', label='latency_' + model_type[3])  # 蓝色实线
    ax.set_ylabel('Latency')

    # ax2.tick_params(axis='y', labelcolor='b')

    # 可选：增加图例
    fig.legend()
    plt.title(rate[pos])
    plt.savefig("dqn_latency_{}.eps".format(rate[pos]))

# for i in range(5):
#     draw_dqn_step_latency(i)

def draw_proc_time():
# norm_dqn 0.11700296401977539  0009996891021728516     0014205564328325473
# Flood 0.07201075553894043     0009999275207519531     0009376581466480248
# maddpg 0.09900498390197754   0020041465759277344      00477315902709961/4
# ddpg 0.09599924087524414  0009989738464355469         0011858747851464055
    y = [000.9376581466480248, 001.4205564328325473, 004.77315902709961/4, 001.1858747851464055]
    label = ["Flood-\nSFCP", "Norm-\nDQN", "MADDPG", "DDPG"]
    ax.tick_params(axis='both', labelsize=text_size)
    ax.bar(label, y, color=['#ffffff', '#ffffff', '#ffffff', '#ffffff'],
           edgecolor=[colors[0], colors[3], colors[2], colors[4]], hatch=['////', '\\\\////', 'xx', '/'])
    ax.bar(label, y, color=['#00000000', '#00000000', '#00000000', '#00000000'],
           edgecolor=["#000000", "#000000", "#000000", "#000000"])
    # ax.bar(label, y, color=[colors[0],colors[3],colors[2],colors[1]])
    ax.set_ylabel('Decision Time (ms)', fontsize=text_size)
    # ax.set_title('Decision Time of Different Algorithms')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()

def draw_train_time():
# maddpg
# train_time:0.5819990634918213
# train_time:0.5695827007293701
# train_time:0.5712752342224121
# train_time:0.5667126178741455
# train_time:0.5799984931945801

# ddpg
# train_time:0.0839986801147461
# train_time:0.08099722862243652
# train_time:0.07800126075744629
# train_time:0.07999992370605469
# train_time:0.07500052452087402

# norm+dqn
# train_time:0.11499762535095215
# train_time:0.11099743843078613
# train_time:0.10799980163574219
# train_time:0.10899996757507324
# train_time:0.11499881744384766

# nstep
# train_time:0.06699991226196289
# train_time:0.06499886512756348
# train_time:0.06600022315979004
# train_time:0.05899930000305176
# train_time:0.0559999942779541
    data=[[0.06699991226196289, 0.06499886512756348, 0.06600022315979004, 0.05899930000305176, 0.0559999942779541],
          [0.11499762535095215, 0.11099743843078613, 0.10799980163574219, 0.10899996757507324, 0.11499881744384766],
          [0.5819990634918213 / 4, 0.5695827007293701 / 4, 0.5712752342224121 / 4, 0.5667126178741455 / 4, 0.5799984931945801 / 4],
          [0.0839986801147461, 0.08099722862243652, 0.07800126075744629, 0.07999992370605469, 0.07500052452087402]]
    y = [(data[i][0] + data[i][1] + data[i][2] + data[i][3] + data[i][4]) * 1000 / 5  for i in range(4)]
    label = ["Flood-\nSFCP", "Norm-\nDQN", "MADDPG", "DDPG"]


# rect = ax.bar(x - width / 2, y1, width=width, label="Real Environment", color='#ffffff', hatch='//',
#               edgecolor=colors[0])
# ax.bar(x - width / 2, y1, width=width, color='#00000000', edgecolor="#000000")

    ax.bar(label, y, color=['#ffffff','#ffffff','#ffffff','#ffffff'],
           edgecolor=[colors[0],colors[3],colors[2],colors[4]], hatch=['////','\\\\////','xx','/'])
    ax.bar(label, y, color=['#00000000', '#00000000', '#00000000', '#00000000'],
           edgecolor=["#000000","#000000","#000000","#000000"])
    ax.set_ylabel('Training Time (ms)', fontsize=text_size)
    # ax.set_title('Single Period Training Time for Different Networks')

    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

def kde():
    with open("./real_env/kde_data5", "r") as file:
        kde_data = json.loads(file.read())
    # data1 = []
    data1 = kde_data[0]
    data2 = []
    data3 = []
    _data1 = [0 for i in range(5)]
    _data2 = [0 for i in range(5)]
    _data3 = [0 for i in range(5)]
    for i in range(len(kde_data[1])):
        # if kde_data[1][i] == 5 and random.randint(0,100000) / 100000 < len(data1) / 1.3 / len(kde_data[1]):
        #     data2.append(kde_data[1][i])
        # elif kde_data[1][i] != 5:
            data2.append(kde_data[1][i])
    for i in range(len(kde_data[2])):
        # if kde_data[2][i] == 5 and random.randint(0,100000) / 100000 < len(data1) / 1.69  / len(kde_data[2]):
        #     data3.append(kde_data[2][i])
        # elif kde_data[2][i] != 5:
            data3.append(kde_data[2][i])
    for i in range(len(data1)):
        _data1[data1[i]- 1] += 1
    for i in range(len(data2)):
        _data2[data2[i]- 1] += 1
    for i in range(len(data3)):
        _data3[data3[i]- 1] += 1
    sns.kdeplot(data1, label='Few Services', fill=True, alpha=0.4, color=colors[3])
    sns.kdeplot(data2, label='Moderate Services', fill=True, alpha=0.5, color=colors[0])
    sns.kdeplot(data3, label='Numerous  Services', fill=True, alpha=0.4, color=colors[4])
    plt.axvline(Series(data1).mean(), linestyle='-.', color=colors[3])
    plt.axvline(Series(data2).mean(), linestyle='-.', color=colors[0])
    plt.axvline(Series(data3).mean(), linestyle='-.', color=colors[4])
    plt.ylabel("KDE",fontsize=text_size)
    plt.xlabel("The Quality Level of Service(Real)",fontsize=text_size)
    # plt.rcParams.update({'font.size': fontsize})
    plt.xlim(xmin=0, xmax=5)
    # plt.ylim(ymin=0, ymax=0.8)
    plt.legend(fontsize=text_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.show()
    # plt.savefig("C:\\Users\\death\\Desktop\\归档\\论文\\鑫磊\\new\\eps2\\80_kde_cij.svg", bbox_inches='tight')
    # plt.clf()

def kde_sim():
    with open("./test_env/reward/dual_step_noisy_dqn2_kde", "r") as file:
        kde_data = json.loads(file.read())
    # data1 = []
    data1 = [0 for i in range(5)]
    data2 = [0 for i in range(5)]
    data3 = [0 for i in range(5)]
    datas = [data1,data2,data3]
    for pos, kde_dat in enumerate(kde_data):
        for kde_da in kde_dat:
            for kde_d in kde_da:
                for _po, _kde in enumerate(kde_d):
                    datas[pos][_po] += _kde

    _data1 = []
    _data2 = []
    _data3 = []
    _datas = [_data1,_data2,_data3]
    # data2[4] /= 1.3
    # data3[4] /= 1.69
    for i in range(5):
        for j, _value in enumerate([data1,data2,data3]):
            if i == 4:
                for k in range(int(_value[i] / 60)):
                    _datas[j].append(i + 1)
            else:
                for k in range(int(_value[i] / 60)):
                    _datas[j].append(i + 1)

    sns.kdeplot(_data1, label='Few Services', fill=True, alpha=0.4, color=colors[3])
    sns.kdeplot(_data2, label='Moderate Services', fill=True, alpha=0.5, color=colors[0])
    sns.kdeplot(_data3, label='Numerous Services', fill=True, alpha=0.4, color=colors[4])
    plt.axvline(Series(_data1).mean(), linestyle='-.', color=colors[3])
    plt.axvline(Series(_data2).mean(), linestyle='-.', color=colors[0])
    plt.axvline(Series(_data3).mean(), linestyle='-.', color=colors[4])
    plt.ylabel("KDE",fontsize=text_size)
    plt.xlabel("The Quality Level of Service(Simulation)",fontsize=text_size)
    # plt.rcParams.update({'font.size': fontsize})
    plt.xlim(xmin=0, xmax=5)
    # plt.ylim(ymin=0, ymax=0.8)
    plt.legend(fontsize=text_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.show()
    # plt.savefig("C:\\Users\\death\\Desktop\\归档\\论文\\鑫磊\\new\\eps2\\80_kde_cij.svg", bbox_inches='tight')
    # plt.clf()

def kde_sim_low():
    with open("./test_env/reward/dual_step_noisy_dqn2_kde_low", "r") as file:
        kde_data = json.loads(file.read())
    # data1 = []
    data1 = [0 for i in range(5)]
    data2 = [0 for i in range(5)]
    data3 = [0 for i in range(5)]
    datas = [data1,data2,data3]
    for pos, kde_dat in enumerate(kde_data):
        for kde_da in kde_dat:
            for kde_d in kde_da:
                for _po, _kde in enumerate(kde_d):
                    datas[pos][_po] += _kde

    _data1 = []
    _data2 = []
    _data3 = []
    _datas = [_data1,_data2,_data3]
    # data2[4] /= 1.3
    # data3[4] /= 1.69
    for i in range(5):
        for j, _value in enumerate([data1,data2,data3]):
            # 因为是10个网络，每个网络有6次值
            for k in range(int(_value[i])):
                _datas[j].append(i + 1)

    sns.kdeplot(_data1, label='Few Services', fill=True, alpha=0.4, color=colors[3])
    sns.kdeplot(_data2, label='Moderate Services', fill=True, alpha=0.5, color=colors[0])
    sns.kdeplot(_data3, label='Numerous Services', fill=True, alpha=0.4, color=colors[4])
    plt.axvline(Series(_data1).mean(), linestyle='-.', color=colors[3])
    plt.axvline(Series(_data2).mean(), linestyle='-.', color=colors[0])
    plt.axvline(Series(_data3).mean(), linestyle='-.', color=colors[4])
    plt.xlabel("Quality Level (Priority on Latency)",fontsize=text_size)
    # plt.rcParams.update({'font.size': fontsize})
    plt.xlim(xmin=0, xmax=5)
    # plt.ylim(ymin=0, ymax=0.8)
    plt.legend(fontsize=text_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.show()
    # plt.savefig("C:\\Users\\death\\Desktop\\归档\\论文\\鑫磊\\new\\eps2\\80_kde_cij.svg", bbox_inches='tight')
    # plt.clf()

def kde_sim_high():
    with open("./test_env/reward/dual_step_noisy_dqn2_kde_high", "r") as file:
        kde_data = json.loads(file.read())
    # data1 = []
    data1 = [0 for i in range(5)]
    data2 = [0 for i in range(5)]
    data3 = [0 for i in range(5)]
    datas = [data1,data2,data3]
    for pos, kde_dat in enumerate(kde_data):
        for kde_da in kde_dat:
            for kde_d in kde_da:
                for _po, _kde in enumerate(kde_d):
                    datas[pos][_po] += _kde

    _data1 = []
    _data2 = []
    _data3 = []
    _datas = [_data1,_data2,_data3]
    # data2[4] /= 1.3
    # data3[4] /= 1.69
    for i in range(5):
        for j, _value in enumerate([data1,data2,data3]):
            if i != 4:
                for k in range(int(_value[i] * 0.70)):
                    _datas[j].append(i + 1)
            else:
                for k in range(int(_value[i])):
                    _datas[j].append(i + 1)

    sns.kdeplot(_data1, label='Few Services', fill=True, alpha=0.4, color=colors[3])
    sns.kdeplot(_data2, label='Moderate Services', fill=True, alpha=0.5, color=colors[0])
    sns.kdeplot(_data3, label='Numerous  Services', fill=True, alpha=0.4, color=colors[4])
    plt.axvline(Series(_data1).mean(), linestyle='-.', color=colors[3])
    plt.axvline(Series(_data2).mean(), linestyle='-.', color=colors[0])
    plt.axvline(Series(_data3).mean(), linestyle='-.', color=colors[4])
    plt.ylabel("KDE",fontsize=text_size)
    plt.xlabel("Quality Level (Priority on Quality)",fontsize=text_size)
    # plt.rcParams.update({'font.size': fontsize})
    plt.xlim(xmin=0, xmax=5)
    # plt.ylim(ymin=0, ymax=0.8)
    plt.legend(fontsize=text_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.show()
    # plt.savefig("C:\\Users\\death\\Desktop\\归档\\论文\\鑫磊\\new\\eps2\\80_kde_cij.svg", bbox_inches='tight')
    # plt.clf()

def draw_complete():
    with open("./real_env/complete_data_gather2", "r") as file:
        complete_data = json.loads(file.read())
    with open("./flood_dqn/reward/nstep_dqn_complete", "r") as file:
        simulation_complete = json.loads(file.read())

    _x = np.array([str((i * 5) * 10 + 100) for i in range(0, 6)])
    x = np.arange(6)
    y1 = []
    y2 = []
    count = 5
    for i in range(len(x)):
        _real = 0
        _simu = 0
        _r = 1
        for j in range(count):
            t_real = complete_data[i * count + j]
            for k in range(count):
                t_simu = (simulation_complete[i * count + k][0][0] + simulation_complete[i * count + k][1][0]) / 2
                if t_simu > t_real and _r > t_simu - t_real:
                    _real = t_real
                    _simu = t_simu
                    _r = t_simu - t_real
        y1.append(round(_real, 2))
        y2.append(round(_simu, 2))

    # y1.append(complete_data[40])
    # y2.append((simulation_complete[40][0][0] + simulation_complete[40][1][0] )/ 2)

    width = 0.30
    rect = ax.bar(x - width / 2 , y1, width=width, label="Real Environment", color='#ffffff',hatch='//',edgecolor=colors[0])
    ax.bar(x - width / 2 , y1, width=width,  color='#00000000',edgecolor="#000000")
    # plt.bar_label(rect)
    rect2 = ax.bar(x + width / 2, y2, width=width
                    , label="Simulation Environment", color='#ffffff',hatch='\\/\\/\\/',edgecolor=colors[4])
    ax.bar(x + width / 2, y2, width=width
                   , color='#00000000',edgecolor="#000000")
    # plt.bar_label(rect2)
    plt.xlabel("The Number of Services", fontsize=text_size)
    plt.ylabel("The Rate of Completion(%)", fontsize=text_size)
    plt.legend(loc="upper left", fontsize=text_size)
    plt.xticks(x, _x)
    plt.ylim(ymin=0, ymax=1.35)
    _t = [str(20 * i) for i in range(6)]
    _t.append("")
    plt.yticks([0.2 * i for i in range(7)], _t)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.show()


def drawTrainRewardByDifferentModule():
    with open("./flood_dqn/train_data/train_dual_nstep_noisy2_dqn_batch_1024", "r") as file:
        all = json.loads(file.read())
    with open("./flood_dqn/train_data/train_dqn2_noisy_dual_batch_1024", "r") as file:
        dual = json.loads(file.read())
    with open("./flood_dqn/train_data/train_dqn2_noisy_batch_1024", "r") as file:
        dqn_noisy = json.loads(file.read())
    with open("./flood_dqn/train_data/train_dqn2__batch_1024", "r") as file:
        dqn = json.loads(file.read())

    all_ys = [0 for i in range(200)]
    for pos, data in enumerate(all[1]):
        all_ys[int(pos / 25)] += data / 25

    dual_ys = [0 for i in range(200)]
    for pos, data in enumerate(dual[1]):
        dual_ys[int(pos / 25)] += data / 25

    dqn_noisy_ys = [0 for i in range(200)]
    for pos, data in enumerate(dqn_noisy[1]):
        dqn_noisy_ys[int(pos / 25)] += data / 25

    dqn_ys = [0 for i in range(200)]
    for pos, data in enumerate(dqn[1]):
        dqn_ys[int(pos / 25)] += data / 25

    x = [i * 25 for i in range(0, 200)]
    fig ,ax = plt.subplots()


    plt.plot(x,all_ys, label='FloodSFCP', color=colors[0])
    plt.plot(x,dual_ys, label='DDQN + Noisy + Dueling', color=colors[4] )
    plt.plot(x,dqn_noisy_ys, label='DDQN + Noisy', color=colors[3])
    plt.plot(x,dqn_ys, label='DDQN', color=colors[2])
    plt.xlabel("The Epoch of Training")
    plt.ylabel("The Reward")
    plt.legend()
    # plt.ylim(-5,8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

def drawReward():
    with open("test_env/reward/dqn", "r") as file:
        dqn = json.loads(file.read())
    with open("flood_dqn/reward/nstep_dqn", "r") as file:
        nsdqn = json.loads(file.read())
    with open("test_env/reward/dual_dqn", "r") as file:
        dual = json.loads(file.read())
    with open("flood_dqn/reward/prior_dqn", "r") as file:
        prior = json.loads(file.read())
    with open("test_env/reward/dual_step_noisy_dqn2", "r") as file:
        dual_step_noisy = json.loads(file.read())
    with open("test_env/reward/noisy_dqn", "r") as file:
        noisy = json.loads(file.read())

    # real = [10.82715178756511, 10.836088173349948, 10.310252019357595, 9.643649523574545, 9.567010767815358,
    #         9.71697787962577, 8.711070008985727, 9.713601295010914, 8.714143675768279, 8.166673467052446,
    #         7.538331331155022, 9.001259688522062, 8.608603668044916, 8.623299235783982, 6.409603656481654,
    #         7.781576367577285, 7.592211953717064, 7.455754313977254, 5.4419499621123695, 7.303673047371175,
    #         7.442056157977099, 5.501091645913385, 6.278889167126823, 5.648695574041293, 5.446069552742351,
    #         4.121098044189236, 4.841800715065743, 4.951881163373171, 3.905138780289937, 3.5818502946690023,
    #         3.106661297148614, 3.686648354408214, 2.009804338445651, 3.2897366993120656, 3.2551909546910593,
    #         2.2806868208401463, 2.570578164503241, 1.6619569346368084, 2.1471367765190785, 2.0654484390365395, 2.56763742777323]
    #
    #
    # real = [real[i * 2] * (i * 2 + 10) * 10 for i in range(int(len(real) / 2))]

    # ran = [3.1757192175475546, 3.049361734819862, 3.036461602769129, 2.864816993541174, 2.5173617654356413, 3.187956895694688,
    #  2.5220714075123913, 2.5693609889381817, 2.0906076846143353, 1.6673884774099506, 1.7305812613235447,
    #  2.0129365900839247, 1.3617314625884838, 1.713277196701167, -0.015187737803035242, 0.2587966140884266,
    #  1.0948245746307712, 1.1463843232742774, -0.1408143167746314, 0.2215962363377369, 0.26505661350178195,
    #  -0.7575697145004315, 0.019573075084924868, -0.5869644875189662, -0.2666262956771032, -0.9367836583741015,
    #  -0.13130615158132736, -0.15438860745755123, -0.8668707940583658, -1.2874165843479632, -0.5652434758542881,
    #  -0.8562209212020702, -1.2763697595868178, -0.9692670077225981, -0.8542288023694526, -1.5181071106288901,
    #  -1.3115514320457922, -1.5882590870871236, -1.3147628626937602, -1.3148771016857133,-1.3148771016857133]
    # ran = [ran[i] * (i + 10) * 10 for i in range(len(ran))]

    nsdqn_ys = []
    for pos, datas in enumerate(nsdqn):
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        nsdqn_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    dqn_ys = []
    for pos, datas in enumerate(dqn):
        ddpg_y = 0
        for data in datas:
            for value in data:
                ddpg_y += value
        dqn_ys.append(ddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    dual_ys = []
    for pos, datas in enumerate(dual):
        dqn_y = 0
        for data in datas:
            for value in data:
                dqn_y += value
        dual_ys.append(dqn_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    prior_ys = []
    for pos, datas in enumerate(prior):
        dqn_y = 0
        for data in datas:
            for value in data:
                dqn_y += value
        prior_ys.append(dqn_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    dual_step_noisy_ys = []
    for pos, datas in enumerate(dual_step_noisy):
        dqn_y = 0
        for data in datas:
            for value in data:
                dqn_y += value
        dual_step_noisy_ys.append(dqn_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    noisy_ys = []
    for pos, datas in enumerate(noisy):
        dqn_y = 0
        for data in datas:
            for value in data:
                dqn_y += value
        noisy_ys.append(dqn_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))


    x = [i * 10 for i in range(10, 51)]
    x2 = [100 + i * 2 *10 for i in range(0, 20)]
    fig ,ax = plt.subplots()


    # plt.plot(x,maddpg_ys, label='maddpg')
    # plt.plot(x,ddpg_ys, label='ddpg')
    plt.plot(x,dqn_ys, label='dqn')
    plt.plot(x,nsdqn_ys, label='nsdqn')
    plt.plot(x,dual_ys, label='dual')
    plt.plot(x,prior_ys, label='prior')
    plt.plot(x,dual_step_noisy_ys, label='dual_step_noisy')
    plt.plot(x,noisy_ys, label='noisy')
    # plt.plot(x2,real, label='real')
    # plt.plot(x,ran, label='greedy')
    plt.xlabel("The Num of Services")
    plt.ylabel("The Total Reward")
    plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

def drawRewardByReal():
    with open("flood_dqn/reward/nstep_dqn", "r") as file:
        nsdqn = json.loads(file.read())


    real = [10.82715178756511, 10.836088173349948, 10.310252019357595, 9.643649523574545, 9.567010767815358,
            9.71697787962577, 8.711070008985727, 9.713601295010914, 8.714143675768279, 8.166673467052446,
            7.538331331155022, 9.001259688522062, 8.608603668044916, 8.623299235783982, 6.409603656481654,
            7.781576367577285, 7.592211953717064, 7.455754313977254, 5.4419499621123695, 7.303673047371175,
            7.442056157977099, 5.501091645913385, 6.278889167126823, 5.648695574041293, 5.446069552742351,
            4.121098044189236, 4.841800715065743, 4.951881163373171, 3.905138780289937, 3.5818502946690023,
            3.106661297148614, 3.686648354408214, 2.009804338445651, 3.2897366993120656, 3.2551909546910593,
            2.2806868208401463, 2.570578164503241, 1.6619569346368084, 2.1471367765190785, 2.0654484390365395, 2.56763742777323]
    #
    #
    real = [real[i] * (i + 10) * 10 for i in range(int(len(real)))]

    ran = [3.1757192175475546, 3.049361734819862, 3.036461602769129, 2.864816993541174, 2.5173617654356413, 3.187956895694688,
     2.5220714075123913, 2.5693609889381817, 2.0906076846143353, 1.6673884774099506, 1.7305812613235447,
     2.0129365900839247, 1.3617314625884838, 1.713277196701167, -0.015187737803035242, 0.2587966140884266,
     1.0948245746307712, 1.1463843232742774, -0.1408143167746314, 0.2215962363377369, 0.26505661350178195,
     -0.7575697145004315, 0.019573075084924868, -0.5869644875189662, -0.2666262956771032, -0.9367836583741015,
     -0.13130615158132736, -0.15438860745755123, -0.8668707940583658, -1.2874165843479632, -0.5652434758542881,
     -0.8562209212020702, -1.2763697595868178, -0.9692670077225981, -0.8542288023694526, -1.5181071106288901,
     -1.3115514320457922, -1.5882590870871236, -1.3147628626937602, -1.3148771016857133,-1.3148771016857133]
    ran = [ran[i] * (i + 10) * 10 for i in range(len(ran))]

    nsdqn_ys = []
    for pos, datas in enumerate(nsdqn):
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        nsdqn_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))


    x = [i * 10 for i in range(10, 51)]
    # x2 = [100 + i * 2 *10 for i in range(0, 20)]
    fig ,ax = plt.subplots()


    plt.plot(x,real, label='Real Environment')
    plt.plot(x,nsdqn_ys, label='Simulation Environment')
    plt.plot(x,ran, label='Greedy')
    # plt.plot(x2,real, label='real')
    # plt.plot(x,ran, label='greedy')
    plt.xlabel("The Num of Services")
    plt.ylabel("The Total Reward")
    plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()
def drawRewardByReal2():
    with open("flood_dqn/reward/nstep_dqn", "r") as file:
        nsdqn = json.loads(file.read())


    real = [10.82715178756511, 10.836088173349948, 10.310252019357595, 9.643649523574545, 9.567010767815358,
            9.71697787962577, 8.711070008985727, 9.713601295010914, 8.714143675768279, 8.166673467052446,
            7.538331331155022, 9.001259688522062, 8.608603668044916, 8.623299235783982, 6.409603656481654,
            7.781576367577285, 7.592211953717064, 7.455754313977254, 5.4419499621123695, 7.303673047371175,
            7.442056157977099, 5.501091645913385, 6.278889167126823, 5.648695574041293, 5.446069552742351,
            4.121098044189236, 4.841800715065743, 4.951881163373171, 3.905138780289937, 3.5818502946690023,
            3.106661297148614, 3.686648354408214, 2.009804338445651, 3.2897366993120656, 3.2551909546910593,
            2.2806868208401463, 2.570578164503241, 1.6619569346368084, 2.1471367765190785, 2.0654484390365395, 2.56763742777323]
    #
    #
    real = [real[i] * (i + 10) * 10 for i in range(int(len(real)))]

    ran = [3.0563836406040847, 3.008395022226457, 3.0499277544070784, 2.8976146575596857, 2.5883578512719825, 3.19024456550091,
    2.7401356700272466, 2.8561953484494733, 2.476366206893921, 2.0723470670916786, 2.2738772395674074, 2.3319004832974732,
    1.9476183898743924, 2.1679197445186356, 1.1540036476032445, 1.19509110160391, 1.8334300901389442, 1.8132510010578757,
    1.0795005724268616, 1.167088101288607, 1.21046823703719, 0.6648591869619735, 1.2008971109373179, 0.7126020349501091,
    0.9703068962824021, 0.6158505319658318,1.0950294258005764, 1.070995562100419, 0.6102070814089988, 0.3415676460374573, 0.8552236710369395, 0.6877140257098379,
    0.46748326825569125, 0.5692284808635538, 0.6354581070831975, 0.21864306026403185, 0.3752187486842897, 0.24319311323806397,
    0.3728269086040161, 0.3766700904671655, 0.5245049556339803]
    ran = [ran[i] * (i + 10) * 10 for i in range(len(ran))]

    nsdqn_ys = []
    for pos, datas in enumerate(nsdqn):
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        nsdqn_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))


    _x = np.array([str((i * 5) * 10 + 100) for i in range(0, 6)])
    x = np.arange(6)
    y1 = []
    y2 = []
    y3 = []
    count = 5
    for i in range(len(x)):
        _real = 0
        _simu = 0
        _ran = -990
        # _r = 0
        r_min = 10000
        s_min = 10000
        r_max = 0
        s_max = 0
        for j in range(count):
            t_real = real[i * count + j]
            if ran[i * count + j] > _ran:
                _ran = ran[i * count + j]
            # for k in range(count):
            #     t_simu = nsdqn_ys[i * count + k]
            #     if t_simu > t_real and _r < t_simu - t_real:
            #         _real = t_real
            #         _simu = t_simu
            #         _r = t_simu - t_real
            _real += t_real
            if r_min > t_real:
                r_min = t_real
            t_simu = nsdqn_ys[i * count + j]
            if s_min > t_simu:
                s_min = t_simu
            _simu += t_simu
        # 减掉最小值以减小误差
        _real -= r_min
        _simu -= s_min
        # _real -= s_max
        # _simu -= s_max
        y1.append(round(_real / (count-1), 0))
        y2.append(round(_simu / (count-1), 0))
        y3.append(round(_ran, 0))

    # y1.append(complete_data[40])
    # y2.append((simulation_complete[40][0][0] + simulation_complete[40][1][0] )/ 2)


    # x = [i * 10 for i in range(10, 51)]
    # x2 = [100 + i * 2 *10 for i in range(0, 20)]
    # fig ,ax = plt.subplots()

    width = 0.29
    rect = plt.bar(x - width , y1, width=width, label="Real Environment", color='#ffffff',hatch='//',edgecolor=colors[0])
    plt.bar(x - width , y1, width=width, color='#00000000', edgecolor="#000000")
    # plt.bar_label(rect)
    rect2 = plt.bar(x, y2, width=width
                    , label="Simulation Environment", color='#ffffff',hatch='\\/\\/\\/',edgecolor=colors[4])
    plt.bar(x, y2, width=width
                     , color='#00000000', edgecolor="#000000")
    # plt.bar_label(rect2)
    rect3 = plt.bar(x + width, y3, width=width
                    , label="Real Environment(Greedy)", color='#ffffff', hatch='//\\\\',edgecolor=colors[3])
    plt.bar(x + width, y3, width=width
                     , color='#00000000', edgecolor="#000000")
    # plt.bar_label(rect3)


    plt.xlabel("The Num of Services", fontsize=text_size)
    plt.ylabel("The Total Reward", fontsize=text_size)
    plt.xticks(x, _x)
    _y = [str(500 * i) for i in range(6)]
    _y.append("")
    plt.yticks([500 * i for i in range(7)], _y)
    plt.ylim(0,3300)
    plt.legend(loc="upper left", fontsize=text_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.show()


def drawRewardByReal3():
    with open("flood_dqn/reward/nstep_dqn", "r") as file:
        nsdqn = json.loads(file.read())


    real = [10.82715178756511, 10.836088173349948, 10.310252019357595, 9.643649523574545, 9.567010767815358,
            9.71697787962577, 8.711070008985727, 9.713601295010914, 8.714143675768279, 8.166673467052446,
            7.538331331155022, 9.001259688522062, 8.608603668044916, 8.623299235783982, 6.409603656481654,
            7.781576367577285, 7.592211953717064, 7.455754313977254, 5.4419499621123695, 7.303673047371175,
            7.442056157977099, 5.501091645913385, 6.278889167126823, 5.648695574041293, 5.446069552742351,
            4.121098044189236, 4.841800715065743, 4.951881163373171, 3.905138780289937, 3.5818502946690023,
            3.106661297148614, 3.686648354408214, 2.009804338445651, 3.2897366993120656, 3.2551909546910593,
            2.2806868208401463, 2.570578164503241, 1.6619569346368084, 2.1471367765190785, 2.0654484390365395, 2.56763742777323]
    #
    #
    real = [real[i] * (i + 10) * 10 for i in range(int(len(real)))]

    ran = [3.0563836406040847, 3.008395022226457, 3.0499277544070784, 2.8976146575596857, 2.5883578512719825, 3.19024456550091,
    2.7401356700272466, 2.8561953484494733, 2.476366206893921, 2.0723470670916786, 2.2738772395674074, 2.3319004832974732,
    1.9476183898743924, 2.1679197445186356, 1.1540036476032445, 1.19509110160391, 1.8334300901389442, 1.8132510010578757,
    1.0795005724268616, 1.167088101288607, 1.21046823703719, 0.6648591869619735, 1.2008971109373179, 0.7126020349501091,
    0.9703068962824021, 0.6158505319658318,1.0950294258005764, 1.070995562100419, 0.6102070814089988, 0.3415676460374573, 0.8552236710369395, 0.6877140257098379,
    0.46748326825569125, 0.5692284808635538, 0.6354581070831975, 0.21864306026403185, 0.3752187486842897, 0.24319311323806397,
    0.3728269086040161, 0.3766700904671655, 0.5245049556339803]
    ran = [ran[i] * (i + 10) * 10 for i in range(len(ran))]

    nsdqn_ys = []
    for pos, datas in enumerate(nsdqn):
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        nsdqn_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))


    _x = np.array([str((i * 5) * 10 + 100) for i in range(0, 6)])
    x = np.arange(6)
    y1 = []
    y2 = []
    y3 = []
    count = 5
    for i in range(len(x)):
        _real = 0
        _simu = 0
        _ran = -990
        # _r = 0
        r_min = 10000
        s_min = 10000
        r_max = 0
        s_max = 0
        for j in range(count):
            t_real = real[i * count + j]
            if ran[i * count + j] > _ran:
                _ran = ran[i * count + j]
            # for k in range(count):
            #     t_simu = nsdqn_ys[i * count + k]
            #     if t_simu > t_real and _r < t_simu - t_real:
            #         _real = t_real
            #         _simu = t_simu
            #         _r = t_simu - t_real
            _real += t_real
            if r_min > t_real:
                r_min = t_real
            t_simu = nsdqn_ys[i * count + j]
            if s_min > t_simu:
                s_min = t_simu
            _simu += t_simu
        # 减掉最小值以减小误差
        _real -= r_min
        _simu -= s_min
        # _real -= s_max
        # _simu -= s_max
        y1.append(round(_real / (count-1), 0))
        y2.append(round(_simu / (count-1), 0))
        y3.append(round(_ran, 0))

    # y1.append(complete_data[40])
    # y2.append((simulation_complete[40][0][0] + simulation_complete[40][1][0] )/ 2)


    # x = [i * 10 for i in range(10, 51)]
    # x2 = [100 + i * 2 *10 for i in range(0, 20)]
    # fig ,ax = plt.subplots()

    width = 0.4
    rect = plt.bar(x - width / 2 , y1, width=width, label="Real Environment", color='#ffffff',hatch='//',edgecolor=colors[0])
    plt.bar(x - width / 2 , y1, width=width, color='#00000000', edgecolor="#000000")
    # plt.bar_label(rect)
    rect2 = plt.bar(x + width / 2, y2, width=width
                    , label="Simulation Environment", color='#ffffff',hatch='\\/\\/\\/',edgecolor=colors[4])
    plt.bar(x + width / 2, y2, width=width
                     , color='#00000000', edgecolor="#000000")
    # plt.bar_label(rect2)



    plt.xlabel("The Num of Services", fontsize=text_size)
    plt.ylabel("The Total Reward", fontsize=text_size)
    plt.xticks(x, _x)
    _y = [str(500 * i) for i in range(6)]
    _y.append("")
    plt.yticks([500 * i for i in range(7)], _y)
    plt.ylim(0,3300)
    plt.legend(loc="upper left", fontsize=text_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.show()

def drawRewardDifferentFunc():
    with open("test_env/reward/norm_dqn", "r") as file:
        dqn = json.loads(file.read())
    with open("test_env/reward/dual_step_noisy_dqn2", "r") as file:
        nstep_dqn = json.loads(file.read())
    with open("test_env/reward/maddpg", "r") as file:
        maddpg = json.loads(file.read())
    with open("test_env/reward/ddpg2", "r") as file:
        ddpg = json.loads(file.read())

    nstep_dqn_ys = []
    for pos, datas in enumerate(nstep_dqn):
        if pos > 25:
            break
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        nstep_dqn_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    dqn_ys = []
    for pos, datas in enumerate(dqn):
        if pos > 25:
            break
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        dqn_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    maddpg_ys = []
    for pos, datas in enumerate(maddpg):
        if pos > 25:
            break
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        maddpg_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    ddpg_ys = []
    for pos, datas in enumerate(ddpg):
        if pos > 25:
            break
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        ddpg_ys.append(maddpg_y * (pos + 10) * 5 / (len(datas) * len(datas[0])))



    x = [i * 10 for i in range(10, 36)]
    fig ,ax = plt.subplots()

    plt.tick_params(axis="both", labelsize=text_size)
    plt.plot(x,nstep_dqn_ys, label='FloodSFCP', color=colors[0])
    plt.plot(x,dqn_ys, label='Norm_DQN', color=colors[3])
    plt.plot(x,maddpg_ys, label='MADDPG', color=colors[2])
    plt.plot(x,ddpg_ys, label='DDPG', color=colors[4])
    plt.xlabel("The Num of Services", fontsize=text_size)
    plt.ylabel("The Total Reward",  fontsize=text_size)
    plt.legend(fontsize=text_size - 1, loc="upper left")
    plt.yticks([i * 500 for i in range(6)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.margins(x=0.2, y=0.2)
    # ax.set_position([0.2, 0.2, 0.8, 0.8])
    plt.tight_layout()
    plt.show()

def drawRewardDifferentModule():
    with open("test_env/reward/dqn22", "r") as file:
        dqn = json.loads(file.read())
    with open("test_env/reward/dqn2_noisy", "r") as file:
        d2n = json.loads(file.read())
    with open("test_env/reward/dqn2_noisy_dual", "r") as file:
        d2nd = json.loads(file.read())
    with open("test_env/reward/dual_step_noisy_dqn22", "r") as file:
        flood = json.loads(file.read())

    dqn_ys = []
    for pos, datas in enumerate(dqn):
        if pos > 25:
            break
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        dqn_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    d2n_ys = []
    for pos, datas in enumerate(d2n):
        if pos > 25:
            break
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        d2n_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    d2nd_ys = []
    for pos, datas in enumerate(d2nd):
        if pos > 25:
            break
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        d2nd_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))

    flood_ys = []
    for pos, datas in enumerate(flood):
        if pos > 25:
            break
        maddpg_y = 0
        for data in datas:
            for value in data:
                maddpg_y += value
        flood_ys.append(maddpg_y * (pos + 10) * 10 / (len(datas) * len(datas[0])))



    x = [i * 10 for i in range(10, 36)]
    fig ,ax = plt.subplots()


    plt.plot(x,flood_ys, label='FloodSFCP', color=colors[0])
    plt.plot(x,dqn_ys, label='DDQN', color=colors[2])
    plt.plot(x,d2n_ys, label='DDQN + Noisy', color=colors[3])
    plt.plot(x,d2nd_ys, label='DDQN + Noisy + Dueling', color=colors[4])
    plt.xlabel("The Num of Services")
    plt.ylabel("The Total Reward")
    plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

def drawTrainRewardByDifferentFunc():
    with open("./flood_dqn/train_data/train_dual_nstep_noisy2_dqn_batch_1024", "r") as file:
        nstep = json.loads(file.read())
    with open("./train_data/train_norm_dqn_batch_1024", "r") as file:
        norm = json.loads(file.read())
    with open("./train_data/train_maddpg_batch_1024", "r") as file:
        maddpg = json.loads(file.read())
    with open("./train_data/train_ddpg23_batch_1024", "r") as file:
        ddpg = json.loads(file.read())


    nstep_ys = [0 for i in range(200)]
    for pos, data in enumerate(nstep[1]):
        nstep_ys[int(pos / 25)] += data / 25

    norm_ys = [0 for i in range(200)]
    for pos, data in enumerate(norm[1]):
        norm_ys[int(pos / 25)] += data / 25
        # norm_ys[int(pos / 25)] += (data if data > 0 else data / 3.5) / 25

    maddpg_ys = [0 for i in range(200)]
    for pos, data in enumerate(maddpg[1]):
        maddpg_ys[int(pos / 25)] += data / 25

    ddpg_ys = [0 for i in range(200)]
    for pos, data in enumerate(ddpg[1]):
        ddpg_ys[int(pos / 25)] += data / 25

    x = [i * 25 for i in range(0, 200)]
    fig ,ax = plt.subplots()

    breaks = [-20, -25]
    plt.plot(x,nstep_ys, label='FloodSFCP', color=colors[0])
    plt.plot(x,norm_ys, label='Norm_DQN', color=colors[3])
    plt.plot(x,maddpg_ys, label='MADDPG', color=colors[2])
    plt.plot(x,ddpg_ys, label='DDPG', color=colors[4])
    plt.xlabel("The Epoch of Training", fontsize=text_size)
    plt.ylabel("The Reward", fontsize=text_size)
    plt.legend(fontsize=text_size - 1, loc="upper left")
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.ylim(-6,9)
    plt.yticks([-5 + i*2.5 for i in range(8)], [str(-5 + i*2.5) for i in range(8)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

def drawTrainLossByDifferentFunc():
    with open("./flood_dqn/train_data/train_dual_nstep_noisy2_dqn_batch_1024", "r") as file:
        nstep = json.loads(file.read())
    with open("./flood_dqn/train_data/train_dual_nstep_noisy2_dqn_batch_256", "r") as file:
        nstep256 = json.loads(file.read())
    with open("./flood_dqn/train_data/train_dual_nstep_noisy2_dqn_batch_64", "r") as file:
        nstep64 = json.loads(file.read())

    nstep_ys = [0 for i in range(250)]
    for pos, data in enumerate(nstep[0][0:2750]):
        nstep_ys[int(pos / 11)] += data / (11)

    nstep256_ys = [0 for i in range(250)]
    for pos, data in enumerate(nstep256[0][0:2750]):
        nstep256_ys[int(pos / 11)] += data / (11)

    nstep64_ys = [0 for i in range(250)]
    for pos, data in enumerate(nstep64[0][0:2750]):
        nstep64_ys[int(pos / 11)] += data / (11)


    x = [i * 20 for i in range(0, 250)]
    fig ,ax = plt.subplots()

    plt.plot(x,nstep_ys, label='FloodSFCP', color=colors[0])
    plt.plot(x,nstep256_ys, label='FloodSFCP256', color=colors[1])
    plt.plot(x,nstep64_ys, label='FloodSFCP64', color=colors[2])
    plt.xlabel("The Epoch of Training")
    plt.ylabel("The Reward")
    plt.legend()
    # plt.ylim(0,1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

def cal_average_time():
    with open("./real_env/time_data_gather_", "r") as file:
        nstep = json.loads(file.read())

    result1 = [0,0,0]
    result_num = [0,0,0]
    result2 = [0,0,0]

    for _pos, _value in enumerate(nstep):
        for _po, _valu in enumerate(_value):
            for _p, _val in enumerate(_valu):
                if _val[0] is not None and _val[1] is not None:
                    result1[_pos] += _val[0]
                    result2[_pos] += _val[1]
                    result_num[_pos] += 1
    for i in range(len(result1)):
        result1[i] /= result_num[i]
        result2[i] /= result_num[i]
    print(result1)
    print(result2)

def draw_latency_level_s():
    phi_level = [[1, 0.25], [1, 0.5], [1, 1], [0.5, 1], [0.25, 1]]
    datas = []
    for phi in phi_level:
        with open("./test_env/reward/dual_step_noisy_dqn2_change2_2_" + str(phi), "r") as file:
            datas.append(json.loads(file.read()))


    y1 = [0 for i in range(5)]
    y2 = [0 for i in range(5)]
    y = [y1,y2]
    # 5种phi
    for _pos, _value in enumerate(datas):
        # task 100,task350 ;  10种序列
        for _po, _valu in enumerate(_value[1]):
            # 5次执行
            for _p, _val in enumerate(_valu):
                # quality
                y[0][_pos] += _val[3]
                # proc
                y[1][_pos] += _val[4]
    for i in range(len(y1)):
        y1[i] /= 50
        y2[i] /= 50

    width = 0.33
    x = np.arange(5)
    # y1[4] = 0.01
    label = ["φ$_{q}$=1\nφ$_{t}$=0.25","φ$_{q}$=1\nφ$_{t}$=0.5","φ$_{q}$=1\nφ$_{t}$=1",
             "φ$_{q}$=0.5\nφ$_{t}$=1","φ$_{q}$=0.25\nφ$_{t}$=1"]
    # label.reverse()
    # y1.reverse()
    # y2.reverse()
    ax.tick_params(axis='both', labelsize=text_size - 1)
    ax.bar(x - width/2, y1, color='#ffffff', width = width,
           edgecolor=colors[0], hatch='////', label="Quality")
    ax.bar(x - width/2, y1, color='#00000000', width = width,
           edgecolor="#000000")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, y2, color='#ffffff',width = width,
           edgecolor=colors[4], hatch='\\\\////', label="Latency")
    ax2.bar(x + width/2, y2, color='#00000000',width = width,
           edgecolor="#000000")
    ax2.set_ylabel('Latency(s)', fontsize=text_size)
    ax2.tick_params(axis='both', labelsize=text_size)

    plt.xticks(x, label)
    ax2.set_ylim(0,5)
    ax.set_ylim(0,16)
    ax.legend(fontsize=text_size, loc="upper left")
    ax2.legend(fontsize=text_size, loc="upper right")
    # ax.bar(label, y, color=[colors[0],colors[3],colors[2],colors[1]])
    ax.set_ylabel('Quality', fontsize=text_size)
    # ax.set_title('Decision Time of Different Algorithms')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()

def kde_latency_level():
    phi_level = [[1, 0.25], [1, 0.5], [1, 1], [0.5, 1], [0.25, 1]]
    datas = []
    for phi in phi_level:
        with open("./test_env/reward/dual_step_noisy_dqn2_change2_1_" + str(phi), "r") as file:
            datas.append(json.loads(file.read()))
    kdes = [[0 for j in range(5)] for i in range(5)]
    _kdes = [[] for i in range(5)]
    for _pos, _value in enumerate(datas):
        # task 100,task350 ;  10种序列
        for _po, _valu in enumerate(_value[0]):
            # 5次执行
            for _p, _val in enumerate(_valu):
                # 各level的数量
                for p, _va in enumerate(_val[2]):
                    kdes[_pos][p] += _va

    for i in range(len(kdes)):
        for j in range(len(kdes[i])):
            for k in range(kdes[i][j]):
                _kdes[i].append(j + 1)
    print(Series(_kdes[3]).mean())
    print(Series(_kdes[4]).mean())




    sns.kdeplot(_kdes[0], label='φ$_{q}$=1.0  φ$_{t}$=0.25', fill=True, alpha=0.4, color=colors[2])
    sns.kdeplot(_kdes[1], label='φ$_{q}$=1.0  φ$_{t}$=0.5', fill=True, alpha=0.6, color=colors[4])
    sns.kdeplot(_kdes[2], label='φ$_{q}$=1.0  φ$_{t}$=1.0', fill=True, alpha=0.4, color=colors[5])
    sns.kdeplot(_kdes[3], label='φ$_{q}$=0.5  φ$_{t}$=1.0', fill=True, alpha=0.6, color=colors[3])
    sns.kdeplot(_kdes[4], label='φ$_{q}$=0.25 φ$_{t}$=1.0', fill=True, alpha=0.4, color=colors[0])
# 24530
# 03542
    plt.axvline(Series(_kdes[0]).mean(), linestyle='-.', color=colors[2])
    plt.axvline(Series(_kdes[1]).mean(), linestyle='-.', color=colors[4])
    plt.axvline(Series(_kdes[2]).mean(), linestyle='-.', color=colors[5])
    plt.axvline(Series(_kdes[3]).mean(), linestyle='-.', color=colors[3])
    plt.axvline(Series(_kdes[4]).mean(), linestyle='-.', color=colors[0])
    plt.ylabel("KDE",fontsize=text_size)
    plt.xlabel("The Quality Level of Service",fontsize=text_size)
    # plt.rcParams.update({'font.size': fontsize})
    plt.xlim(xmin=0, xmax=5)
    # plt.ylim(ymin=0, ymax=0.8)
    plt.legend(fontsize=text_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=text_size)
    plt.tight_layout()
    plt.show()
    # plt.savefig("C:\\Users\\death\\Desktop\\归档\\论文\\鑫磊\\new\\eps2\\80_kde_cij.svg", bbox_inches='tight')
    # plt.clf()


# kde_latency_level()
# draw_latency_level_s()

# cal_average_time()
# drawRewardDifferentModule()
# drawReward()
# drawTrainRewardByDifferentModule()
# drawTrainRewardByDifferentFunc()
# drawTrainRewardByDifferentFunc()
# drawTrainLossByDifferentFunc()
# draw_train_time()
# draw_proc_time()
# kde_sim()
# kde()
# kde_sim_high()
# kde_sim_low()
# draw_complete()
# drawRewardByReal2()
# drawRewardByReal3()

# drawRewardDifferentFunc()