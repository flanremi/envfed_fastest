import json
import os
import random
import shutil


# 各种工具函数

def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames

def create_data():
    # client拼错了，将错就错吧
    file_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}.txt"
    data_url = "./data_25/{}/client{}.data"
    model_url="./model_25/{}/client{}/"
    epochs = [16,32,64,128,256]
    type_names = ["crossing", "high_way", "main_road", "total"]
    with open("./model.data", "r") as file:
        tmp = file.read()
    for type_name in type_names:
        if not os.path.exists("./data_25/{}".format(type_name)):
            os.mkdir("./data_25/{}".format(type_name))
        for pos in range(25):
            _data_url = data_url.format(type_name, pos)
            _tmp = tmp
            _tmp = _tmp.format(model_url.format(type_name, pos), epochs[pos % len(epochs)],
                        "None", file_url.format(type_name, str(pos) + "_train"),
                        file_url.format(type_name, str(pos) + "_val"))
            with open(_data_url, "w") as f:
                f.write(_tmp)


# 一次性函数，用来整合img为fast要求的形式
def create_img():
    file_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}\\"
    # val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}_val\\"
    # txt_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\"
    type_names = ["crossing", "high_way", "main_road", "total"]
    suffixs1 = ["", "","_val", "_val"]
    suffixs2 = ["\\images", "\\labels","\\images", "\\labels"]
    for type_name in type_names:
        for pos in range(25):
            for i in range(len(suffixs1)):
                files = get_all_filenames(file_url.format(type_name, str(pos) + suffixs1[i] + suffixs2[i]))
                for file in files:
                    shutil.move(file, file_url.format(type_name, str(pos) + suffixs1[i]))
                os.removedirs(file_url.format(type_name, str(pos) + suffixs1[i] + suffixs2[i]))

# 生产train.txt和val.txt
def create_txt():
    file_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}"
    type_names = ["crossing", "high_way", "main_road", "total"]
    for type_name in type_names:
        for pos in range(25):
            files = get_all_filenames(file_url.format(type_name, str(pos)))
            files_val = get_all_filenames(file_url.format(type_name, str(pos) + "_val"))
            with open(file_url.format(type_name, str(pos) + "_train.txt"), "w") as f:
                tmp = ""
                for file in files:
                    if file.find("labels.cache") == -1 and file.find(".txt") == -1:
                        tmp += file + "\n"
                tmp = tmp[:-1]
                f.write(tmp)
            with open(file_url.format(type_name, str(pos) + "_val.txt"), "w") as f:
                tmp = ""
                for file in files_val:
                    if file.find("labels.cache") == -1 and file.find(".txt") == -1:
                        tmp += file + "\n"
                tmp = tmp[:-1]
                f.write(tmp)


def get_val_data():
    file_url = "./model_25/{}/client{}/val_result"
    type_names = ["crossing", "high_way", "main_road", "total"]
    val_data = []
    for type_name in type_names:
        _val_data = []
        for pos in range(25):
            with open(file_url.format(type_name, pos), "r") as file:
                tmp = json.loads(file.read())
                if len(tmp) < 256:
                    tmp.extend([tmp[-1] for i in range(256 - len(tmp))])
            _val_data.append(tmp)
        val_data.append(_val_data)
    with open("model_val_data", "w") as file:
        file.write(json.dumps(val_data))


def get_25_lantency():
    print([random.randint(1000, 6000) / 10000 for i in range(25)])

# 制作测试集文件
def create_val_data():
    file_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\{}\\clinet{}"
    type_names = ["high_way"]
    tmp = ""
    tmp_val = ""
    for type_name in type_names:
        for pos in range(25):
            files = get_all_filenames(file_url.format(type_name, str(pos)))
            files_val = get_all_filenames(file_url.format(type_name, str(pos) + "_val"))
            for file in files:
                if file.find("labels.cache") == -1 and file.find(".txt") == -1:
                    tmp += file + "\n"
            for file in files_val:
                if file.find("labels.cache") == -1 and file.find(".txt") == -1:
                    tmp_val += file + "\n"

    tmp_val = tmp_val[:-1]
    tmp = tmp[:-1]
    with open("./dqn_data", "w") as file:
        file.write(tmp)
    with open("./dqn_val_data", "w") as file:
        file.write(tmp_val)

# ========================================================
def gatherAllData():
    dir_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\crossing\\clinet{}\\"
    target_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\now_crossing\\"
    for i in range(25):
        for t_url in ["", "_val"]:
            a = dir_url.format(str(i) + t_url)
            files = get_all_filenames(a)
            for file in files:
                if file.find("labels.cache") == -1:
                    shutil.move(file, target_url)


def gather_val():
    val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\now_crossing\\client{}_val\\"
    dir_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\now_crossing\\client{}\\"
    for cli in range(10):
        if not os.path.exists(val_url.format(cli)):
            os.mkdir(val_url.format(cli))
        for i in range(36):
            files = get_all_filenames(dir_url.format(cli))
            pos = random.randint(0, int(len(files)/2) - 1)
            shutil.move(files[pos * 2], val_url.format(cli))
            shutil.move(files[pos * 2 + 1], val_url.format(cli))

def split_client():
    dir_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\now_crossing\\"
    files = get_all_filenames(dir_url)
    for i in range(10):
        target_url = dir_url + "client" + str(i) + "\\"
        os.mkdir(target_url)
        l = int(len(files) / 20)
        for j in range(l):
            shutil.move(files[(l * i  + j) * 2], target_url)
            shutil.move(files[(l * i  + j) * 2 + 1], target_url)
# 制作新的测试集文件
def create_now_data():
    val_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\now_crossing\\client{}_val\\"
    dir_url = "C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\clients\\now_crossing\\client{}\\"
    for i in range(10):
        with open("./model.data", "r") as file:
            tmp = file.read()
        tmp = tmp.format("./new_model_25/client" + str(i) + "/", i % 3 + 1,
                    "./modelzoo/coco2017-0.241078ap-model.pth",
                    dir_url.format(i),
                    val_url.format(i))
        with open("./new_data_25/client{}.data".format(i), "w") as f:
            f.write(tmp)

create_now_data()
# getherAllData()
# get_val_data()
# create_txt()
# create_data()