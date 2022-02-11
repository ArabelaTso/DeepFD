import os
import sys
import subprocess
sys.path.append("./")

print(os.path.abspath(os.path.curdir))
classifierLists = os.listdir("Classifiers")
classifierLists = ["All"]
for classifier in classifierLists:
    subprocess.call("python CodeBook/MultiLabelClassification/predictClassifier.py -pd=Benchmark -ds=Subjects -md=Classifiers/{} -thr=0".format(classifier), shell=True)