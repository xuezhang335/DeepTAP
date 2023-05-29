# DeepTAP

DeepTAP is a novel recurrent neural network (RNN)-based, using bidirectional gated recurrent unit (BiGRU), developed for the accurate prediction of TAP-binding peptides.The DeepTAP web server is freely accessible to the public at https://pgx.zju.edu.cn/deeptap/. 

Contact: zhanzhou@zju.edu.cn

# Download and installation

### System

Windows/Linux

### Dependencies
Create a virtual environment using conda or miniconda. This is the version of the package that must be installed.
 see the **env.yaml** file for detailed installation packages and versions.

|Packages|
|----|
|python==3.10.0|
|numpy==1.24.2|
|pandas==1.5.3|
|pytorch==1.1.3|
|torchmetrics=0.11.1|
|pytorch-lightning==1.9.2|

### Steps

Download the latest version of DeepTAP from https://github.com/zjupgx/DeepTAP or https://github.com/xuezhang335/DeepTAP

    git clone https://github.com/zjupgx/DeepTAP.git
    (or git clone https://github.com/xuezhang355/DeepTAP.git)

Go into the directory by using the following command:

    cd DeepTAP

# General usage


### Input files

DeepTAP takes **csv** or **xlsx** files as input with head of **"peptide"** (requisite). See **demo/demo1.csv** for an example.

|peptide|
|----|
|KADDDKPGA|
|PTAWRSEMN|
|AEASAAAAY|
|KKTSLEKRM|
|AAASAAYAY|
|RRFGDKLNF|
|ALAKAGAAV|
|AAASAAAAK|
|ALAAAAAAQ|

### Parameters
-t, --taskType, choices=['cla', 'reg'], Select task type: classification, regression<br>
-p, --peptide, Single peptide for prediction<br>
-f, --file, Input file with peptides for prediction: if given, overwrite -p option<br>
-o, --outputDir, Directory to store file with prediction result: if not given, the current directory will be applied

### Running
Single peptide:

    classification model prediction:
        python deeptap.py -t cla -p LNIMNKLNI -o <output directory>
    regression model prediction:
        python deeptap.py -t reg -p LNIMNKLNI -o <output directory>

List of peptides in a file:

    classification model prediction:
        python deeptap.py -t cla -f <input file> -o <output directory>
    regression model prediction:
        python deeptap.py -t reg -f <input file> -o <output directory>

### Output
The model prediction results output two files: the original ranking file and the ranking file according to the prediction score from high to low.
See **demo/demo1_DeepTAP_cla_predresult_rank.csv** for an example.
|peptide|pred_score|pred_label|
|----|----|----|
|AAASAAYAY|0.9955|1|
|RRFGDKLNF|0.9884|1|
|AEASAAAAY|0.9795|1|
|AAASAAAAK|0.6483|1|
|KKTSLEKRM|0.5658|1|
|ALAKAGAAV|0.3405|0|
|KADDDKPGA|0.2964|0|
|ALAAAAAAQ|0.2531|0|
|PTAWRSEMN|0.1114|0|

The following are the field descriptions for the result file.<br>
For classification tasks:<br>
pred_score: Combined prediction score, between 0-1, the threshold is 0.5<br>
pred_label: Whether it is a binding peptide, 0 means no binding, 1 means binding<br>


For regression tasks:<br>
pred_affinity: Binding prediction affinity, unit nM, threshold is 10000nM<br>
pred_label: Whether it is a binding peptide, 0 means no binding, 1 means binding<br>

# Update log

V1.0  
Test the suitabilty of different RNN variants (GRU,LSTM,BGRU,BLSTM,att-BGRU and att-BLSTM) and CNN on the binding prediction and select the best one (BGRU) for model construction.


