import argparse
import os
import sys
print(os.path.abspath(os.curdir))
sys.path.append('./')
from collections import defaultdict
from CodeBook.Config import FAULT_TYPE
import json
from CodeBook.Utils.isKilled import is_diff_sts
from CodeBook.Utils.analysis_utils import parse_fault_str, convert_bool2int, extract_feature, has_enough_feature, \
    has_enough_sample
from CodeBook.Utils.FileHandler import validate_path, read_csv, find_files
import pandas as pd


def get_acc(result, output_file="stats.csv"):
    acc_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for res in result:
        if "raw_data" in str(res):
            # debug only
            # print("raw_data:", res, res.parent.parent)
            continue
        fault_name = res.parent.parent.parent
        model_name = fault_name.parent
        dataset_name = model_name.parent

        with open(res, "r") as f:
            last_acc = f.readlines()[-1].rstrip("\n").split(',')[1]
            last_acc = float(last_acc)
            acc_dict[str(dataset_name)][str(model_name)][str(fault_name)].append(last_acc)

    for dataset, dvalues in acc_dict.items():
        if dataset == "RQ2":
            print("Skip")
            continue

        print("Dataset:", dataset)

        for model, mvalues in dvalues.items():
            print("Model:", model)

            max_iter = 0
            content = ""

            for fault, fvalues in mvalues.items():
                print("Fault:", fault)
                print(fvalues)
                if len(fvalues) < 20:
                    print("\n!!! Insufficient iteration, skip.")
                    continue

                fault_list, _ = parse_fault_str(str(fault).split("/")[-1])

                fault_str = ",".join(fault_list)
                content += str(fault) + "," + fault_str + "," + ",".join([str(fv) for fv in fvalues]) + "\n"
                max_iter = max(max_iter, len(fvalues))

            # open a file to write
            output_path = os.path.join(model, output_file)
            fw = open(output_path, 'w')
            print("Output statistic result to:", output_path)
            fw.write("Fault_dir,{},{}\n".format(",".join(FAULT_TYPE),
                                                ",".join(["iter_{}".format(x) for x in range(1, max_iter + 1)])))
            fw.write(content)
            fw.close()

    # return acc_dict


def cal_is_kill(stats_list, last=20, output_file="stats_faulty.csv"):
    total_fault_cnt = 0
    for stat in stats_list:
        # debug only
        print("\nProcess", str(stat))

        with open(stat, "r") as f:
            lines = f.readlines()

        title = lines[0].strip("\n")
        original_line = ""
        other_lines = []

        origin_acc_list = []
        other_acc_lists = []
        cur_fault_cnt = 0

        for line in lines[1:]:
            line = line.strip("\n")
            parts = line.split(",")
            cur_acc_list = parts[-last:]

            # debug only
            # print("cur_acc_list", cur_acc_list)

            fault_type = parts[0].split("/")[-1]
            if fault_type == "origin":
                origin_acc_list = [float(v) for v in cur_acc_list]
                original_line = line
            else:
                other_acc_lists.append([float(v) for v in cur_acc_list])
                other_lines.append(line)
        if not len(origin_acc_list):
            # "origin" folder doesn't exist in this model, then skip this model
            print("Cannot find 'origin' in {}, skip.".format(str(stat)))
            continue

        # debug only
        # print("origin_acc_list", origin_acc_list)
        # print("other_acc_lists", other_acc_lists)

        is_faulty_list = []
        new_other_lines = []
        ave_origin_acc = sum(origin_acc_list) / len(origin_acc_list)

        for other_acc, other_line in zip(other_acc_lists, other_lines):
            # a model is faulty if :
            # - it is statistically different from the original model and
            # - its accuracy is less than the original's
            ave_cur_acc = sum(other_acc) / len(other_acc)

            is_kill = is_diff_sts(origin_acc_list, other_acc)
            is_faulty = int(is_kill and ave_cur_acc < ave_origin_acc)

            other_line += ",{},{},{}".format(ave_cur_acc, is_kill, is_faulty)
            new_other_lines.append(other_line)
            # count how many mutants have been killed and faulty
            if is_faulty:
                cur_fault_cnt += 1
            is_faulty_list.append(str(is_faulty))

        total_fault_cnt += cur_fault_cnt
        # debug only
        print("is_faulty_list:", is_faulty_list)
        print("Current faulty: ", cur_fault_cnt)

        print("Total faulty: ", total_fault_cnt)

        # write to file
        with open(os.path.join(stat.parent, output_file), "w") as f:
            f.write(title + ",ave_score,is_killed,is_faulty\n")
            f.write(original_line + ",{},0,0\n".format(ave_origin_acc))
            f.write("\n".join(new_other_lines))


def summary(parent_dir, is_subject, max_iter=20):
    dataset_summary_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    is_faulty_cnt = 0
    is_killed_cnt = 0

    for model_dir in os.listdir(parent_dir):
        # model_dir should not be a file
        if os.path.isfile(os.path.join(parent_dir, model_dir)):
            continue

        print("\nModel dir: ", model_dir)
        acc_dict = defaultdict(list)
        # list all mutants
        for mutant_dir in os.listdir(os.path.join(parent_dir, model_dir)):
            # if mutant dir == "raw_data" or not a directory, then skip
            if mutant_dir == "raw_data" or os.path.isfile(os.path.join(parent_dir, model_dir, mutant_dir)):
                continue

            print("\nMutant dir: ", mutant_dir)
            # validate sufficient iter
            existing_iterations = [f for f in os.listdir(os.path.join(parent_dir, model_dir, mutant_dir)) if
                                   f.startswith("iter_")]
            if len(existing_iterations) < max_iter:
                print("  - Insufficient iter {} !".format(len(existing_iterations)))
                print("  - Skip this mutant! {}\n".format(os.path.join(parent_dir, model_dir, mutant_dir)))
                break
            else:
                print("  - Sufficient iter, continue.")

            # get seeded faults and labels
            num_fault = len(mutant_dir.split("__")) if mutant_dir != "origin" else 0
            fault_list, label_dict = parse_fault_str(mutant_dir)

            # collect accuracy of 20 iters
            acc_list = []
            for iter_num in range(1, max_iter + 1, 1):
                # print("iter: ", iter_num)
                log_dir = os.path.join(parent_dir, model_dir, mutant_dir, "iter_{}".format(iter_num), "log_dir",
                                       "log.csv")
                feature_dir = os.path.join(parent_dir, model_dir, mutant_dir, "iter_{}".format(iter_num), "result_dir",
                                           "monitor_features.csv")
                autotrainer_dir = os.path.join(parent_dir, model_dir, mutant_dir, "iter_{}".format(iter_num),
                                               "result_dir", "monitor_detection.log")
                # validate file exist
                validate_path(log_dir)
                validate_path(feature_dir)
                validate_path(autotrainer_dir)

                # get fault number, get faults and labels
                dataset_summary_dict[model_dir][mutant_dir][iter_num]["num_fault"] = num_fault
                for ftype, fvalue in zip(FAULT_TYPE, fault_list):
                    dataset_summary_dict[model_dir][mutant_dir][iter_num]["vl_{}".format(ftype)] = fvalue
                    dataset_summary_dict[model_dir][mutant_dir][iter_num]["lb_{}".format(ftype)] = label_dict[ftype]
                    # for debug
                    # print("dataset_summary_dict[model_dir][mutant_dir][iter_num][lb_{}] = {}".format(ftype,
                    #                                                                                  label_dict[ftype]))

                # ################################################
                #   log.csv
                # val_loss,val_accuracy,loss,accuracy
                with open(log_dir, "r") as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        dataset_summary_dict[model_dir][mutant_dir].pop(iter_num)
                        print("\n{} is empty! Record removed.\n".format(log_dir))
                        continue
                    heads = lines[0].strip('\n').split(',')
                    values_last_line = lines[-1].strip("\n").split(",")

                    # update dataset_summary_dict
                    for head, value in zip(heads, values_last_line):
                        dataset_summary_dict[model_dir][mutant_dir][iter_num]["ft_{}".format(head)] = value
                    # get val_accuracy and add into acc_list
                    var_acc = values_last_line[3]
                    acc_list.append(float(var_acc))

                # ################################################
                # monitor_detection.csv
                # checktype,current_epoch,issue_list,time_usage,Describe
                with open(autotrainer_dir, "r") as f:
                    lines = f.readlines()
                    if len(lines) < 2:
                        dataset_summary_dict[model_dir][mutant_dir].pop(iter_num)
                        print("\n{} is empty! Record removed.\n".format(autotrainer_dir))
                        continue
                    total_time = float(lines[-1].strip("\n").split(",")[-2])
                    ave_time = total_time / (len(lines) - 1)
                    autoTrainer_identified = lines[-1].strip("\n").split(",")[-1] != "No Issue now"

                    # print("last line: {}".format(lines[-1]))
                    # print("Average time = {} / {} = {}, autoTrainer: {}".format(total_time, len(lines) - 1, ave_time,
                    #                                                             autoTrainer_identified))

                    dataset_summary_dict[model_dir][mutant_dir][iter_num]["time"] = ave_time
                    dataset_summary_dict[model_dir][mutant_dir][iter_num][
                        "autoTrainer"] = "1" if autoTrainer_identified else "0"

                # ################################################
                # get features
                df = read_csv(feature_dir)
                df = df.fillna(0.0)

                # if has_enough_feature(df, min_feature=10) and has_enough_sample(df, min_sample=5):
                # preprocess, convert bool dtype ot int if necessary
                try:
                    df = convert_bool2int(df)
                except AttributeError as ae:
                    dataset_summary_dict[model_dir].pop(mutant_dir)
                    print("\n{} is empty! Record removed.\n".format(feature_dir))
                    continue

                feature_dict = extract_feature(df)
                for feat_key, feat_val in feature_dict.items():
                    dataset_summary_dict[model_dir][mutant_dir][iter_num][feat_key] = feat_val

                # debug only
                # print("    - Iter {} finished.".format(iter_num))

            # validate and add to dict
            # print("len(acc_list) = {}".format(len(acc_list)))
            assert len(acc_list) >= max_iter
            # if len(acc_list) < max_iter:
            #     print("acc_list not enough! {}".format(len(acc_list)))

            acc_dict[mutant_dir] = acc_list

        # print("  - Collect Accuracy list finished.")
        # debug only
        # print("{}'s accuracy list: {}".format(mutant_dir, acc_list))

        if is_subject:
            continue
        # validate "origin" exists, if not, delete info for this model
        if "origin" not in acc_dict:
            print("\n\nWarning!!!Insufficient iteration in {}".format(os.path.join(model_dir, "origin")))
            if model_dir in dataset_summary_dict:
                dataset_summary_dict.pop(model_dir)
            continue

        # calculate mutation killing
        print("  - Start Calculating Mutation killing...")
        ave_acc_origin = sum(acc_dict["origin"]) / len(acc_dict["origin"])
        for mut, acc_l in acc_dict.items():
            ave_acc_mut = sum(acc_l) / len(acc_l)
            is_kill = int(is_diff_sts(acc_dict["origin"], acc_l))
            is_faulty = int(is_kill and (ave_acc_mut < ave_acc_origin))

            is_killed_cnt += is_kill
            is_faulty_cnt += is_faulty

            # print(
            #     "ave_acc_mut: {} | ave_acc_origin: {} | is_kill: {} | is_faulty: {}".format(ave_acc_mut, ave_acc_origin,
            #                                                                                 is_kill, is_faulty))

            impact_val_acc = ave_acc_origin - ave_acc_mut

            # update dataset_summary_dict
            for iter_num in range(1, 21, 1):
                dataset_summary_dict[model_dir][mut][iter_num]["is_kill"] = is_kill
                dataset_summary_dict[model_dir][mut][iter_num]["is_faulty"] = is_faulty
                dataset_summary_dict[model_dir][mut][iter_num]["impact_val_acc"] = "%.6f" % impact_val_acc

                # if not faulty, change labels to all zeros
                if not is_faulty:
                    for ftype in FAULT_TYPE:
                        dataset_summary_dict[model_dir][mut][iter_num]["lb_{}".format(ftype)] = 0
                        dataset_summary_dict[model_dir][mut][iter_num]["num_fault"] = 0

        print("  - Mutation killing finished.")
        print("  - Current is_killed_cnt: {}, is_faulty_cnt: {}".format(is_killed_cnt, is_faulty_cnt))

    # for debug only
    # print("\n\n[dataset_summary_dict]")
    # for model_dir in dataset_summary_dict:
    #     print("Model: {}".format(model_dir))
    #     for mutant_dir in dataset_summary_dict[model_dir]:
    #         print(" - Mutant: {}".format(mutant_dir))
    #         for iter_num in dataset_summary_dict[model_dir][mutant_dir]:
    #             print("  - Iter: {}".format(iter_num))
    #             for k, v in dataset_summary_dict[model_dir][mutant_dir][iter_num].items():
    #                 print("\t-{}: {}".format(k, v))

    return dataset_summary_dict


def dict2csv(dataset_summary_dict, output_dir):
    # transfer dict to csv
    # print("dataset_summary_dict", dataset_summary_dict)
    reformed_summary_dict = {}

    for model_dir in dataset_summary_dict:
        print("Model: {}".format(model_dir))
        for mutant_dir in dataset_summary_dict[model_dir]:
            print(" - Mutant: {}".format(mutant_dir))
            for iter_num in dataset_summary_dict[model_dir][mutant_dir]:
                # print("  - Iter: {}".format(iter_num))
                for k, v in dataset_summary_dict[model_dir][mutant_dir][iter_num].items():
                    reformed_summary_dict[(model_dir, mutant_dir, iter_num)] = \
                        dataset_summary_dict[model_dir][mutant_dir][iter_num]

    df = pd.DataFrame.from_dict(reformed_summary_dict, orient="index")
    df.to_csv(output_dir)
    print("Output to {}".format(output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate accuracy.')
    parser.add_argument('--program_dir', '-pd', default='./RQ2', help='program path')
    parser.add_argument('--dataset', '-ds', default='MNIST', help='dataset path')
    parser.add_argument('--target', '-tgt', default='log.csv', help='target filename')
    parser.add_argument('--output_prefix', '-prefix', default='summary', help='prefix of output filename')
    parser.add_argument('--stat', '-stat', default=0, choices=[0, 1], type=int, help="Calculate stats.csv or not.")
    parser.add_argument('--last', '-lst', default=20, type=int, help="Last n accuracy.")
    parser.add_argument('--diff', '-diff', default=0, choices=[0, 1], type=int, help="Calculate stats_diff.csv or not.")
    parser.add_argument('--overwrite', '-ov', default=0, choices=[0, 1], type=int, help="Overwrite or not.")
    parser.add_argument('--recalc', '-rc', default=0, choices=[0, 1], type=int, help="Recalculate or not.")
    parser.add_argument('--max_iter', '-iter', default=20, type=int, help="The iteration that counts.")
    parser.add_argument('--is_subject', '-subj', default=False, type=bool, help="whether working on real subject")

    # set up
    args = parser.parse_args()
    program_dir = args.program_dir
    dataset_dir = args.dataset
    target_filename = args.target
    max_iter = args.max_iter

    stat_file = "stats.csv"
    diff_file = "stats_faulty.csv"
    prefix = args.output_prefix
    summary_csv_file = "{}.csv".format(prefix)
    summary_json_file = "{}_dict.json".format(prefix)
    summary_file_path = os.path.join(program_dir, dataset_dir, summary_json_file)

    if not args.stat:
        if args.recalc or (not os.path.exists(summary_file_path)):
            print("Calculate summary.csv")
            # get summary of features and labels
            summary_dict = summary(os.path.join(program_dir, dataset_dir), args.is_subject, 
                                   max_iter=max_iter)  # {"a":{"b":{"c":{"d":"e"}}}}
            print("summary_dict", summary_dict)

            prev_summary_dict = {}

            if not args.overwrite and os.path.exists(summary_file_path):
                with open(summary_file_path, 'r') as fr:
                    try:
                        prev_summary_dict = json.load(fr)
                    except Exception as e:
                        print("Load json from {} failed because {}".format(summary_file_path, e))

            summary_dict.update(prev_summary_dict)
            with open(summary_file_path, 'w') as fw:
                json.dump(summary_dict, fw)
        else:
            print("Load summary from {}.".format(summary_file_path))
            with open(summary_file_path, 'r') as fr:
                summary_dict = json.load(fr)

        dict2csv(summary_dict, os.path.join(program_dir, dataset_dir, summary_csv_file))

    # ####################################
    # below calculate "stat.csv" in each mutant
    # calculate each model
    if args.stat:
        # find all dirs
        result = find_files(target_filename, os.path.join(program_dir, dataset_dir))

        print("Found {} {} in {}/{}".format(len(result), target_filename, program_dir, dataset_dir))
        #
        get_acc(result, stat_file)

    if args.diff:
        stats_list = find_files(stat_file, os.path.join(program_dir, dataset_dir))
        cal_is_kill(stats_list, last=args.last, output_file=diff_file)
