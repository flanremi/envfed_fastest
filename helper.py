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
    print([random.randint(1000, 10000) / 10000 for i in range(25)])

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



# get_val_data()
# create_txt()
# create_data()