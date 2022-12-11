# Predictive-Process-Monitoring

## Intra-case
1. Download the preprocessed logs from [here](https://drive.google.com/drive/u/0/folders/1PHnikHtH2shAK9LHujmrvPqWTXTAxqkS).
2. Navigate to the Intra-case folder. 

### Training
Make the following changes in train.py for each log:  
  1. Set the paths to the train, validation and test splits at lines 45-52.  
  2. Set the value of maxPrefixLength variable located above where the `train_model` function is called to the corresponding value from MaxPrefixLength.md.
  3. Set the path to output the weights where the `train_model` function is called.

Run `python3 train.py`.

### Testing 
Make the following changes in evaluate.py for each log:  
  1. Set the paths to the train, validation and test splits at lines 47-54.  
  2. Set the value of maxPrefixLength variable located above where the `load_state_dict` function is called to the corresponding value from MaxPrefixLength.md.
  3. Set the path to read the weights where the `load_state_dict` function is called.

Run `python3 evaluate.py`

## Inter-case
1. Download the datasets from [here]( https://drive.google.com/drive/folders/1KQu-jIJqooRsZTgakhOJ0eoXX2z23aR_?usp=sharing).
2. Navigate to the Inter-case folder. 

### Training
Navigate to the Train folder and make the following changes in the file corresponding to the dataset for each sub-dataset:  
  1. Set the path to the dataset at line 12.
  2. For the BPIC 2015 and Hospital Billing datasets, the variable dataset_name needs to be set depending on which sub-dataset is being used since the preprocessing for the sub-datasets is slightly different.  
  3. Set the value of maxPrefixLength variable located above where the `train_model` function is called to the corresponding value from MaxPrefixLength.md. Note that for the datasets that have only one sub-dataset this value may still not be set to the correct value so please check it.
  4. Set the path to output the weights where the `train_model` function is called.  

Run `python3 {filename.py}` where filename.py is the name of the file. 

### Testing 
Navigate to the Test folder and make the following changes in the file corresponding to the dataset for each sub-dataset:  
  1. Set the path to the dataset at line 13.
  2. For the BPIC 2015 and Hospital Billing datasets, the variable dataset_name needs to be set depending on which sub-dataset is being used since the preprocessing for the sub-datasets is slightly different.  
  3. Set the value of maxPrefixLength variable located above where the `load_state_dict` function is called to the corresponding value from MaxPrefixLength.md. Note that for the datasets that have only one sub-dataset this value may still not be set to the correct value so please check it.
  4. Set the path to read the weights where the `load_state_dict` function is called.

Run `python3 {filename.py}` where filename.py is the name of the file. 
