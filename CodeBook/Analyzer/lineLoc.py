import sys
from CodeBook.Analyzer.ASTParser import *
import os
import ast
import numpy as np

sys.path.append("./")


def lineLoc(program_dir, dataset_dir, pred_voting, label_name, subject_name):
    # input predicted label, subject names
    # shape of pred: [5], shape of subject_name: str
    # label_name = "lb_opt, lb_lr, lb_act, lb_loss, lb_epoch"
    label_name = ['optimizer', 'lr', 'act', 'loss', 'epoch']
    base_dir = os.path.join(program_dir, dataset_dir) #"./Evaluations/Subjects"
    subject_dir = os.path.join(base_dir, str(subject_name))
    analyzer = Analyzer()
    for mutant_dir in os.listdir(subject_dir):
        if os.path.isfile(os.path.join(subject_dir, mutant_dir)):
            continue
        content = open(os.path.join(subject_dir, mutant_dir, "origin.py")).read()
        tree = ast.parse(content)
        try:
            analyzer.visit(tree)
            line_no = analyzer.lineno  # line_no: dict["optimizer":[x,x], "loss":[x,x], ...]
            buggy_label = np.where(pred_voting == 1)[0]
            print("Diagnosed / Localized by DeepFD:")
            for lid, label_index in enumerate(buggy_label, 1):
                name = label_name[label_index]

                print("Fault {}: [{}], Lines: {}".format(lid, name, line_no[name]))

        except Exception as e:
            print(e)
            continue
