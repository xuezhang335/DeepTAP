# DeepTAP

TAPPred is a deep learning approach used for predicting high-confidence TAP-binding peptide.

Contact: zhanzhou@zju.edu.cn

# Download and installation

## Git (All the dependencies should be properly installed)

### System

Windows/Linux

### Dependencies

python
pytorch

### Steps

Download the latest version of DeepTAP from https://github.com/xuezhang335/DeepTAP

    [git clone https://github.com/zhangxue355/DeepTAP.git](https://github.com/xuezhang335/DeepTAP.git)

Go into the directory by using the following command:

    cd DeepTAP

Invoke the setup script:

    python setup.py install

# General usage

Single peptide:

    classification model prediction:
    	deeptap -t cla -p <LNIMNKLNI> -o <output directory>
    regression model prediction:
        deeptap -t reg -p <LNIMNKLNI> -o <output directory>

List of peptides in a file:

    classification model prediction:
    	deeptap -t cla -f <input file> -o <output directory>
    regression model prediction:
        deeptap -t reg -f <input file> -o <output directory>

# Input files

DeepTAP takes **csv** files as input with head of **"peptide"** (requisite). For example (demo/demo1.csv)

# Update log

## 2022.03

V1.0  
Test the suitabilty of different RNN variants (GRU,LSTM,BGRU,BLSTM,att-BGRU and att-BLSTM) and CNN on the binding prediction and select the best one (BGRU) for model construction.
