import math
import numpy as np
import os, glob, json, argparse

def format_labeltxt(path, char_num=30000):
    label_path = path + "/label.txt"
    #label_path = os.path.expanduser("~/Pictures/dataset/ocr/label.txt")
    f = open(label_path, "w", encoding="utf-8")
    for i, json_file in enumerate(glob.glob(path + "/auto*.txt")):
        if i >= char_num and char_num > 0:
            break
        if i != 0 and i % 1000 == 0:
            print("%s lines has processed"%(i))
        with open(json_file, "r", encoding="utf-8") as file:
            img_name = json_file[json_file.rfind("/") + 1:json_file.rfind("_")] + ".png"
            try:
                data = json.load(file)
            except:
                print(json_file)
            bbox = []
            n = 0
            while True:
                try:
                    bbox.append(data["characters"][str(n)]["box"])
                    n += 1
                except KeyError:
                    break
            line = "%s:%s:%s%s" % (img_name, "".join(data["string-formatted"]), str(bbox), "\n")
            f.write(line)

def divide_txt(path, name):
    file_path = os.path.join(path, name)
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()
        for i, line in enumerate(content):
            if i % 10000==0:
                label = os.path.join(path+"_0%s"%(int(i/10000)), "label.txt")
                print(label)
                f = open(label, "w", encoding="utf-8")
            f.write(line)

def clean_words(path, delete, replace):
    label_file = os.path.join(path, "label.txt")
    new_label_file = open(os.path.join(path, "new_label.txt"), "w", encoding="utf-8")
    with open(label_file, "r", encoding="utf-8") as file:
        content = file.readlines()
        for i,line in enumerate(content):
            break_line = False
            label = line[line.find(":")+1: line.rfind(":")]
            coord = line[line.rfind(":")+1:]
            name = line[:line.find(":")]
            for char in delete:
                if label.find(char)!=-1:
                    #os.system("rm %s"%(os.path.join(path, name)))
                    #print("%s has been removed"%(os.path.join(path, name)))
                    break_line = True
                    break
            if break_line:
                continue
            for key in replace.keys():
                label = label.replace(key, replace[key])
            #print("%s:%s:%s"%(name, label, coord))
            new_label_file.write("%s:%s:%s"%(name, label, coord))

def verify(path, char_num):
    if not os.path.exists(path):
        print("%s does not exists!"%(path))
        return
    with open(path, "r", encoding="utf-8") as label:
        content = label.readlines()
        print("label.txt under folder: ")
        if char_num != len(content):
            print("%s has %s missing lines" % (path, char_num - len(content)))
            print("Please check the error if possible")
        else:
            print("%s has %s lines"%(path, len(content)))
            print("No problem occurs!")

def get_param():
    parser = argparse.ArgumentParser( description='')
    parser.add_argument('--root', type=str, default="~/workspace/",
                        help='root of the dataset')
    parser.add_argument('--char_num', type=int, default=10000,
                        help='num')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_param()
    for i in range(10):
        path = os.path.expanduser("/dl_data/wang/li_data_0%s"%(i))
        #divide_txt(path, "label_all.txt")
        #format_labeltxt(path, char_num=args.char_num)
        clean_words(path, delete=["･"], replace={"\\": "¥"})
        #verify(path + "/label.txt", args.char_num)
