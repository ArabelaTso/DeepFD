import os
import shutil
import argparse


# from SeedFaults.seedFaults import FAULTS
FAULTS = {
    "loss": ["categorical_crossentropy", "binary_crossentropy",
             "mean_absolute_error", "mean_squared_error"],
    "epoch": [2, 5, 10],
    "lr": [1e-5, 1e-10, 1., 2., 3.]
}


def flaten_faults(faults_dict: dict):
    faults_list = []
    cnt = 0
    for k, vals in faults_dict.items():
        for val in vals:
            faults_list.append("{}_{}".format(k, val))
            cnt += 1
    print("Total faults: ", cnt)
    return faults_list


def checker(path, fault_list=[], exclude_origin=True, verbose=False):
    total = 0
    folder_index = 1
    for prog_folder in os.listdir(path):
        if not os.path.isdir(os.path.join(path, prog_folder)):
            continue

        folder_list = []
        for child_folder in os.listdir(os.path.join(path, prog_folder)):
            if not os.path.isdir(os.path.join(path, prog_folder, child_folder)):
                continue
            elif os.path.exists(os.path.join(path, prog_folder, child_folder, "result_dir", "checkpoint_model")):
                # if no model saved, it also cannot be counted
                if len(os.listdir(os.path.join(path, prog_folder, child_folder, "result_dir", "checkpoint_model"))) > 0:
                    if exclude_origin and child_folder == "origin":
                        continue
                    folder_list.append(child_folder)
                else:
                    # remove child folder which has no saved checkpoint
                    print("Empty Checkpoint found!", prog_folder, child_folder)
                    shutil.rmtree(os.path.join(path, prog_folder, child_folder))
        total += len(folder_list)
        # because 1 is origin, so if any fault has been seeded, the number of child folder should be at least 2
        # if len(folder_list) >= 2:
        #     print(prog_folder)

        # -1 is because there is an "origin" folder which is not faulty program
        print(folder_index, prog_folder, len(folder_list))
        folder_index += 1

        if verbose:
            report_progress(folder_list, fault_list)

    print("Total generated faulty program(s):", total)


def report_progress(finished, total, verbose=False):
    ok_list = []
    miss_list = []
    for item in total:
        if item in finished:
            ok_list.append(item)
        else:
            miss_list.append(item)

    progress = len(total) / len(ok_list)

    print("Progress: total: {}, done: {} ({}%), miss: {}.".format(len(total), len(ok_list), progress, len(miss_list)))
    if verbose:
        print("Missed: ")
        print(",".join(miss_list))
    return miss_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progress Checker')
    parser.add_argument('--root', '-rt', default="../Programs/", help="programs root path")
    parser.add_argument('--dataset', '-ds', default='MNIST', help='dataset path')
    parser.add_argument('--exclude_origin', '-exo', default=0, help='exclude origin or not.', choices=[0, 1], type=int)

    args = parser.parse_args()

    root_path = args.root
    data_name = args.dataset
    exo = args.exclude_origin

    folder_path = os.path.join(root_path, data_name)

    fault_list = flaten_faults(FAULTS)
    fault_list.append("origin")
    checker(folder_path, fault_list, exclude_origin=exo)
