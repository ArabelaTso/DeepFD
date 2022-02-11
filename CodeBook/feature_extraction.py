import os
import sys
import subprocess
sys.path.append("./")

# calculate status for each case
subprocess.call("python CodeBook/Utils/cal_stats.py -pd=Benchmark -ds=Subjects -iter=10 -subj=True -ov 1 -stat=1", shell=True)
# extract for all cases
subprocess.call("python CodeBook/Utils/cal_stats.py -pd=Benchmark -ds=Subjects -iter=10 -subj=True -stat=0", shell=True)

