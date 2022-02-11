import os
import shutil
import subprocess
from pathlib import Path
import pandas as pd

def run_all(dir):
    for seed_folder in os.listdir(dir):
        print("Handling ", seed_folder)

        model_path, config_path = "", ""

        files = os.listdir(os.path.join(dir, seed_folder))
        for file in files:
            if file.endswith("h5"):
                model_path = os.path.join(dir, seed_folder, file)
            elif file.endswith("pkl"):
                config_path = os.path.join(dir, seed_folder, file)
        try:
            process_status = subprocess.run(
                ['python', './run.py', '-mp', model_path, '-cp', config_path, '-sf', '1'],
                check=True)
            print(process_status)

        except subprocess.CalledProcessError as e:
            print("{} failed".format(seed_folder))
            print(e)


def find_file_by_suffix(search_dir, suffix=".csv"):
    if not os.path.isdir(search_dir):
        print("Invalid path!", search_dir)
        return None

    files = list(
        filter(lambda x: os.path.isfile(os.path.join(search_dir, x)) and x.endswith(suffix), os.listdir(search_dir)))
    files.sort(key=lambda x: os.path.getmtime(os.path.join(search_dir, x)))

    return files


def find_files(filename, search_path):
    result = []
    # Walking top-down from the root
    for root, dir, files in os.walk(search_path):
        # print(dir)
        if filename in files:
            result.append(Path(os.path.join(root, filename)))
    return result


def validate_path(dir):
    if not os.path.exists(dir):
        raise FileNotFoundError("File Not Found! {}".format(dir))


def remove_folder_if_log_notexist(search_path):
    for root, dirs, files in os.walk(search_path):
        # print("root=[{}] | dirs = [{}] | files = [{}]".format(root, dirs,files))
        if root.split("/")[-1] == "log_dir" and ("log.csv" not in files):
            print("Log file not found in", root)
            print("Parent", Path(root).parent)
            shutil.rmtree(Path(root).parent)


def read_csv(file_path: str):
    df = None
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print("Read {} Failed.".format(file_path))
        print(e)
    return df

# if __name__ == '__main__':
#     prog_path = "RQ2"
#     datasets = ["MNIST", "MNIST2", "CIFAR-10", "Circle", "Blob"]  # , "IMDB", "Reuters"
#     for dataset in datasets:
#         remove_folder_if_log_notexist(os.path.join(prog_path, dataset))
