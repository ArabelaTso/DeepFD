# DeepFD

DeepFD, a **learning-based** fault diagnosis and localization framework which **maps the fault localization task to a learning problem**. 

In particular, it infers the **suspicious fault types** via monitoring the runtime features extracted during DNN model training, and then locates the diagnosed faults in DL programs.



## Workflow of DeepFD

![workflow](./Figures/workflow.png)



### Step 1: Diagnostic Feature Extraction

Given a program, DeepFD constructs a DNN architecture and collects runtime data such as the loss and neurons' information by training the architecture. 



### Step 2: Fault Diagnosis

After obtaining the diagnostic features, we then **infer the possible types of faults** that exist in the DL program according to the features. We regard it as a **multi-label classification problem**, which maps the obtained features into one or more possible labels.  Each label corresponds to a fault type. 

The classification relies on the predictions made by a set of **pre-trained diagnosis models** (see below). The diagnosis result is given by the union of the diagnosed faults predicted by each diagnosis model to maximize the number of faults diagnosed.



### Step 3: Fault Localization

After acquiring the diagnosed types of faults, DeepFD performs fault localization at the program level. Specifically, the program is first parsed into its abstract syntax tree (AST), and DeepFD goes through the nodes of the parse tree, traverses assignment statement as well as expressions, and then identifies the places (i.e. lines) where the diagnosed types of faults are defined. 



## Diagnosis Model Construction

![model-prep](./Figures/model-prep.png)

### Step 1: Fault Seeding. 

This step is to prepare sufficient training samples for the diagnosis models.



### Step 2: Training Set Preparation

This step prepares the labels and extracting inputs for the training set.



### Step 3: Diagnosis Model Training

We treat the fault diagnosis as a multi-label classification problem mapping a faulty program to the multiple types of faults that it contains. We construct three diagnosis models using the three widely-used and effective machine learning algorithms (i.e., K-Nearest Neighbors, Decision Tree and Random Forest ) to learn the correlations between diagnostic features and types of faults.



## Quick Start

### Prerequisite

- Python 3.6 +
- Packages:

```shell
pip install -r requirements.txt
```



### Fault Diagnosis and Localization

- Run the script

```shell
python CodeBook/predict.py
```



- Example output

```shell
====================
Working on case: 
working on subject:  59282996
The ground truth faults: ['epoch']
Diagnosed / Localized by DeepFD:
Fault 1: [epoch], Lines: [309]
case end 
====================


====================
Working on case: 
working on subject:  31556268
The ground truth faults: ['lr', 'loss', 'epoch']
Diagnosed / Localized by DeepFD:
Fault 1: [lr], Lines: [17]
Fault 2: [act], Lines: [11, 13]
Fault 3: [loss], Lines: [19]
Fault 4: [epoch], Lines: [34]
case end 
====================

```



## Benchmark

- Under the folder `Benchmark/Subjects`, there are two subjects.

- The statistics, patches, detailed information of all benchmarks can be found at `Benchmark/Subjects/Benchmark-with-patches.xlsx`
- The evaluation statistics can be found at ``Benchmark/Subjects/Evaluation.xlsx`
- The entire benchmark with runtime information and extracted features can be downloaded [here](https://drive.google.com/drive/folders/1JddnrUPy1cqM5XAVwAye6aeE-FVl_qoa?usp=sharing). 



## Other Functions

### Feature Extraction from Runtime Information

- Script

```shell
python CodeBook/feature_extraction.py
```

- Output
  - `summary.csv` under folder (specified by `-pd`)
  - `summary_dict.csv` under folder (specified by `-pd`)
  - `stats.csv` in each sub-folder (specified by `-ds`)



### Fault Seeding

- Prepare original dataset
  - Download from dataset provided [here](https://conf.researchr.org/details/icse-2021/icse-2021-papers/81/AUTOTRAINER-An-Automatic-DNN-Training-Problem-Detection-and-Repair-System).
  - Download into folder `Evaluation`, sub-folders inside including `MNIST`, `CIFAR-10`, etc.

- Generate mutant

```shell
python CodeBook/seed_all.py --base Evaluation --dataset MNIST -sf 1 --fault_type loss ----max_fault 2
```

The example seeds 2 faults in type of `loss` for each DNN model under `Evaluation/MNIST`.



- Run all the mutants and collect runtime information

```shell
python CodeBook/run_all.py --base Evaluation --dataset MNIST --gpu 0 --run 1 --max_iter 1
```

The example runs each generated mutant once, together with original one under `Evaluation/MNIST`, without using GPU.



- Extract features and labeling

```shell
python CodeBook/feature_extraction.py
```



## Project Structure

```shell
DeepFD
├── Classifiers  # The pretrained diagnosis models
│   └── All
├── CodeBook     # code
│   ├── Analyzer
│   ├── Callbacks
│   ├── Config.py
│   ├── MultiLabelClassification
│   ├── SeedFaults
│   ├── Utils
│   ├── feature_extraction.py
│   ├── predict.py
│   ├── run_all.py
│   └── seed_all.py
├── Benchmark   # Benchmarks
│   └── Subjects
├── Figures			# Figures for README
│   ├── model-prep.png
│   └── workflow.png
├── README.md		
└── requirements.txt  # requirements to be installed by pip
```



