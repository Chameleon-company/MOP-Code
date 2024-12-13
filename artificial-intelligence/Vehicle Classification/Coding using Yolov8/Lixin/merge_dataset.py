import os
import shutil
now_path_MOP_Code = os.getcwd()
print(now_path_MOP_Code)
now_path_Vehicle_Classification = os.path.join(now_path_MOP_Code, "artificial-intelligence", "Vehicle Classification")
print(now_path_Vehicle_Classification)
now_path_Coding_using_Yolov8 = os.path.join(now_path_Vehicle_Classification, "Coding using Yolov8")
print(now_path_Coding_using_Yolov8)
now_path_MyTest = os.path.join(now_path_Coding_using_Yolov8, "MyTest")
dataset_path = os.path.join(now_path_Vehicle_Classification, "dataset")
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

def create_dataset_dir():
    dir_list = ["train", "val"]
    appendix = ["images", "labels"]
    for app in appendix:
        for dir in dir_list:
            dir_path = os.path.join(dataset_path, app)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            dir_path = os.path.join(dir_path, dir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    # for dir in dir_list:
    #     dir_path1 = os.path.join(dataset_path, "images")
    #     dir_path2 = os.path.join(dataset_path,"labels")

    #     if not os.path.exists(dir_path1):
    #         os.makedirs(dir_path1)
    #     if not os.path.exists(dir_path2):
    #         os.makedirs(dir_path2)
        
    #     images_path = os.path.join(dir_path1, dir)
    #     if not os.path.exists(images_path):
    #         os.makedirs(images_path)
    #     labels_path = os.path.join(dir_path2, dir)
    #     if not os.path.exists(labels_path):
    #         os.makedirs(labels_path)

create_dataset_dir()



def merge_dataset():
    appendix = {}
    exception = ["ethanlongmuir-twoaxlebus-013.jpg","ethanlongmuir-fiveaxleartic-157.jpg",]
    for dir in os.listdir(now_path_Vehicle_Classification):
        if dir.startswith("vehicle") and dir[-1].isdigit():
            now_dir_list = ["val", "train"]
            copy_list = ["images", "labels"]
            copy_list2 = ["images", "labels"]
            for now_dir in now_dir_list:
                for i,copy in enumerate(copy_list):
                    src_dir = os.path.join(now_path_Vehicle_Classification, dir, now_dir, copy_list2[i])
                    dst_dir = os.path.join(dataset_path,copy,now_dir, )
                    print(src_dir, dst_dir)
                    for file in os.listdir(src_dir):
                        append = file.split(".")[1]
                        if append == "webp" or file in exception:
                            continue
                        if append not in appendix:
                            appendix[append] = 1
                        else:
                            appendix[append] += 1
                        src_file = os.path.join(src_dir, file)
                        dst_file = os.path.join(dst_dir, file)
                        shutil.copyfile(src_file, dst_file)
    print(appendix)
                    
                
merge_dataset()