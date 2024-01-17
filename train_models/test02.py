import os


def get_files_in_current_folder(folder_path="."):
    # 获取当前文件夹下所有文件的路径
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    return files


def change_file_extension(directory, old_extension, new_extension):
    for filename in os.listdir(directory):
        if filename.endswith(old_extension):
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, os.path.splitext(filename)[0] + new_extension)
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} to {os.path.basename(new_file_path)}')


if __name__ == "__main__":
    # 指定文件夹路径，默认为当前文件夹
    folder_path = "/home/yy/data/wildfire/flames2/labels"
    # 获取当前文件夹下所有文件的路径
    old_extension = '.jpg'
    new_extension = '.txt'

    change_file_extension(folder_path, old_extension, new_extension)
    # # 打印文件路径
    # for file_path in files_in_folder:
    #     print(file_path)
