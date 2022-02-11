import ast
import subprocess
import astunparse
import os
import json
import shutil
import sys
sys.path.append("../")
from CodeBook.Config import ACT
from CodeBook.Utils.save_pkl import *


def traverse(case_path, dest_path, overwrite=False, run_py=False):
    files = os.listdir(case_path)
    for file in files:
        # only handle python files
        if not file.endswith("py"):
            continue

        # derive post id, new a folder naming after it
        post_id = file.split(".")[0]
        post_folder = os.path.join(dest_path, post_id)

        # if the path exists and overwrite argument is set to be False, then move on to the next post
        if os.path.exists(post_folder) and os.path.exists(os.path.join(post_folder, "origin.h5")) and not overwrite:
            # print(post_id + " handled, move on the next.")
            continue
        else:
            print("Handling ", file)

        # make new dir
        os.makedirs(post_folder, exist_ok=True)

        # copy the original buggy code to the folder
        shutil.copy(os.path.join(case_path, file), os.path.join(post_folder, "origin.py"))

        # read the content
        content = open(os.path.join(case_path, file)).read()

        output_path = None
        try:
            # parse and analyze
            output_path = parse_analysis(content, post_folder)
        except Exception as e:
            print("{} failed".format(post_id))
            print(e)

        # If want to run each program, uncomment the following
        if run_py:
            if output_path is not None and os.path.exists(output_path):
                try:
                    process_status = subprocess.run(['python', output_path], check=True)
                # try:
                #     process_status.check_returncode()
                except subprocess.CalledProcessError as e:
                    print("{} failed".format(post_id))
                    print(e)


def parse_analysis(content, post_folder):
    tree = ast.parse(content)

    # for debug only
    # print(ast.dump(tree))

    # traverse AST
    analyzer = Analyzer()
    analyzer.visit(tree)

    # save collected configuration
    analyzer.report_config(save_path=os.path.join(post_folder, "config_gen.json"))
    analyzer.report_lineno(save_path=os.path.join(post_folder, "lineno.json"))

    # handle body
    stmt_gen_model = gen_save_model_stmt(analyzer.get_model_id(), post_folder, "origin.h5")

    # for debug only
    # print("body_list", len(body_list))

    # insert import
    tree.body.insert(0, ast.parse("from keras.models import load_model").body[0])
    tree.body.insert(len(tree.body), ast.parse(stmt_gen_model).body[0])

    # unparse the AST to source code, and write to
    source_code = astunparse.unparse(tree)
    output_path = os.path.join(post_folder, "origin_ins.py")
    with open(output_path, "w") as fw:
        fw.write(source_code)

    # run the code
    # exec(compile(tree, filename="", mode="exec"))
    return output_path


def add_import(imports):
    new_imp = ast.parse("from keras.models import load_model")
    imports.append(new_imp)
    return list(set(imports))


def insert_stmt(node, index, stmt):
    node.body[0].insert(index, ast.parse(stmt).body[0])


def gen_save_model_stmt(model_id, path, filename):
    return "{}.save(\"{}\")".format(model_id, str(os.path.join(path, filename)))


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.model_id = "model"
        self.configs = {"batch_size": None,
                        "epoch": None,
                        "loss": None,
                        "lr": None,
                        "optimizer": None,
                        "act": []}
        self.lineno = {"batch_size": [],
                       "epoch": [],
                       "loss": [],
                       "lr": [],
                       "optimizer": [],
                       "act": []
                       }

    def visit_Assign(self, node):
        # print("\nAssign:", node.__dict__)
        self._handle_assign_exp(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        # print("\naugAssing", node.__dict__)
        self._handle_assign_exp(node)

    def visit_Expr(self, node):
        # print("\nExp: ", node, node.__dict__)
        self._handle_assign_exp(node)

    def _handle_assign_exp(self, node):
        # handle assignment and expressions
        for n in ast.iter_child_nodes(node):
            # print("_handle_assign_exp", n.__dict__)
            if 'lineno' not in n.__dict__:
                continue
            lineno = n.__dict__['lineno']
            for f in n._fields:
                child = n.__dict__[f]
                if isinstance(child, list) and len(child):
                    for ch in child:
                        self._handle_each(ch, lineno)
                else:
                    self._handle_each(child, lineno)

    def _handle_each(self, ch, lineno):
        # handle each child of a given node
        if isinstance(ch, ast.keyword):
            self._handle_keyword(ch, lineno)
        elif isinstance(ch, ast.Call):
            self._handle_call(ch)

    def _handle_call(self, node):
        # handle Call
        call_name = ""
        # print("_handle_call", node.__dict__)
        # if "Activation" in node.__dict__["func"].id:
        #     self.lineno['act'].append(node.__dict__["lineno"])
        lineno = node.__dict__["lineno"]

        func, args, keywords = node.func, node.args, node.keywords
        if isinstance(func, ast.Attribute):
            call_name = self._handle_attr(func)
        elif isinstance(func, ast.Name):
            call_name = func.id
        for arg in args:
            self._handle_arg(arg)
        for kw in keywords:
            self._handle_keyword(kw, lineno)
        return call_name

    def _handle_attr(self, attribute):
        # print("_handle_attr", attribute.__dict__)
        value = ""
        if attribute.attr == "fit":
            print("find fit")
            self.model_id = attribute.value.id
            value = self.model_id
        if isinstance(attribute.value, ast.Str):
            value = attribute.value.s
        if isinstance(attribute.value, ast.Name):
            value = attribute.value.id
        return value

    def _handle_arg(self, argument):
        # print("_handle_arg", argument.__dict__)
        if 's' in argument.__dict__ and argument.__dict__['s'] in ACT:
            self.configs['act'].append(argument.__dict__['s'])
            self.lineno["act"].append(argument.__dict__['lineno'])
        if not isinstance(argument, ast.arg):
            return
        if isinstance(argument, ast.Num):
            print(argument.n)
        elif isinstance(argument, ast.Call):
            self._handle_call(argument)

    def _handle_keyword(self, kw, lineno):
        if not isinstance(kw, ast.keyword):
            return

        # print("_handle_keyword", kw.__dict__)
        para, val = None, None
        if isinstance(kw, ast.keyword):
            para = kw.arg
            if isinstance(kw.value, ast.Num):
                val = kw.value.n
            elif isinstance(kw.value, ast.Str):
                val = kw.value.s
            elif isinstance(kw.value, ast.Name):
                val = kw.value.id
            elif isinstance(kw.value, ast.Call):
                val = self._handle_call(kw.value)
            self._search_config_in_assign(para, val, lineno)

    def _search_config_in_assign(self, para_name, value, lineno):
        # search training configuration from assignment by keywords
        if value is None:
            return
        if isinstance(value, ast.Call):
            self._handle_call(value)
        # print("find in", para_name, value, type(para_name), type(value))
        if "epochs" in para_name or "nb_epoch" in para_name:
            self.configs["epoch"] = value
            self.lineno["epoch"].append(lineno)
        if "lr" in para_name or "learning_rate" in para_name:
            self.configs["lr"] = value
            self.lineno["lr"].append(lineno)
        if "batch_size" in para_name:
            self.configs["batch_size"] = value
            self.lineno["batch_size"].append(lineno)
        if "loss" in para_name:
            self.configs["loss"] = value
            self.lineno["loss"].append(lineno)
        if "optimizer" in para_name:
            self.configs["optimizer"] = value
            self.lineno["optimizer"].append(lineno)
        if "act" in para_name:
            self.configs["act"].append(value)
            self.lineno["act"].append(lineno)

    def report_config(self, save_path=None):
        # print info
        print("\n{}\n[Configuration]".format("=" * 20))
        for conf, val in self.configs.items():
            print("{}: {}".format(conf, val))
        print("\n{}\n[End of Configuration]".format("=" * 20))

        # save detected config to path
        if save_path is not None:
            with open(save_path, 'w') as fp:
                json.dump(self.configs, fp)

            # save pkl
            config = pack_train_config(opt=self.configs["optimizer"], loss=self.configs["loss"], dataset="",
                                       epoch=self.configs["epoch"], batch_size=self.configs["batch_size"],
                                       lr=self.configs["lr"])
            save_config(config,
                        os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split(".")[0] + ".pkl"))

    def report_lineno(self, save_path=None):
        # print info
        print("\n{}\n[Lineno]".format("=" * 20))
        for conf, lineno in self.lineno.items():
            print("{}: {}".format(conf, lineno))
        print("\n{}\n[End of Lineno]".format("=" * 20))

        # save detected config to path
        if save_path is not None:
            with open(save_path, 'w') as fp:
                json.dump(self.lineno, fp)

    def get_model_id(self):
        return self.model_id


if __name__ == "__main__":
    # traverse("../Cases", "../TryCase", overwrite=True, run_py=False)

    # test one case
    content = open(os.path.join("../Cases", "48385830.py")).read()
    tree = ast.parse(content)

    # for debug only
    # print(ast.dump(tree))

    # traverse AST
    analyzer = Analyzer()
    analyzer.visit(tree)
    analyzer.report_config()
    analyzer.report_lineno()
