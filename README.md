#CAC:Confidence-Aware Co-training Methods for Weakly Supervised Crack Segmentation 

​A Pytorch implementation of Weakly Supervised Crack Segmentaion projects.

1. Datasets：Crack500, CrackForest, DeepCrack. <br>
Notes：please download the corresponding dataset and prepare it by following the guidance.
2. Installation：
You can create a new Conda environment using the command:

``` 
conda env create -f environment.yml
```
3. Training：
- Before the training, please download the dataset and copy it into the folder "datasets".
--datasets
----crack500
----CrackForest
----DeepCrack
- Check the hyperparameters of CAC training in ./options/base_options.py and ./options/train_options.py.
- Training CAC model by meta_train_with_crack500.py
``` 
python meta_train_with_crack500.py
```
4. Testing:
- Check the hyperparameters of CAC testing in ./options/base_options.py and ./options/test_options.py.
``` 
python test_meta_with_crack500.py
```
Notes: the testing dataset name can be replaced in python file test_meta_with_crack500.py.
5. Evaluation：

``` 
cd eval
python eval.py --metric_mode prf --model_name crack500_CAM_proportion --output crack500_CAM_proportion.prf --f1_threshold_mode ois
```

