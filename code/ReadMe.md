# Source code of "Anomaly Detection of Defect using Energy of Point Pattern Features within Random Finite Set Framework"

## Install anaconda env
```conda env create -f environment.yml```


```conda activate rfs```

## Download the D2-net features of MVTec AD dataset from [Here](https://drive.google.com/file/d/1VfY_8HXRwi8_UeTwHrpq6-2lPxrZ6JB_/view?usp=sharing) and place them under d2-net-features directory


### Optinal:Change the directory of the D2_Net features at the configuration file for each category in './configs/


## Run the following for training and evaluation 

```python  MVTecAD_train_evaluation_github.py --d2-net-features-path './d2-net-features/```
