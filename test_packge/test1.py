import os

def get_files_in_current_folder(folder_path="."):
    # 获取当前文件夹下所有文件的路径
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, file))]
    return files


if __name__ == "__main__":
    file_path = get_files_in_current_folder(folder_path="/home/yy/data/wildfire/flames2/temp")
    for name in file_path:
        txt_path = "/home/yy/data/wildfire/flames2/labels/" + name.split("/")[-1].split(".")[0]
        with open(txt_path, 'w') as f:
            f.write("")

