# CostCert

This is the code for CostCert.

## Environment

The code is implemented in Python==3.8, timm==0.9.10, torch==2.0.1.

## Datasets

- [ImageNet](https://image-net.org/download.php) (ILSVRC2012)
- [CIFAR-10/CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/)

## Files

├── train_drs.py              #Training for voting-based recovery base model 

├── certification_drs.py                  #Evaluate voting-based recovery defender

├── topkcert.py    #Compute the certification of top k for peers and CostCert

├── topkcert_readcsv.py    #Read the results save by  topkcert.py

├── train_model.py              #Training for masking-based recovery base model (for extended CostCert)

├── pc_certification.py                  #Evaluate masking-based recovery defender (for extended CostCert)

├── drs.pc_readcsv.py                  #Read the results save by  topkcert.py and compute with the masking one to get the extended CostCert (for extended CostCert)

## Demo

0. You may need to configure the location of datasets and checkpoints.

1. First, train base DL models. 

  ```python
  `python train_drs.py --dataset gtsrb --ablation_type column --model vit_base_patch16_224 --ablation_size 19
  ```

  

2. Then, get the inference results of samples in the dataset from the DL models.

  ```python
  `python certification_drs.py --dataset gtsrb --ablation_type column --model vit_base_patch16_224 --ablation_size 19
  ```

  

3. Finally, get the results of TopKCert.

  ```python
  `python topkcert.py --dataset gtsrb --ablation_type column --model vit_base_patch16_224 --ablation_size 19
  ```



4. You may run [topkcert_readcsv.py](https://github.com/anonymoussubmissio/CostCert/blob/main/topkcert_readcsv.py) or [drs+pc_readcsv.py](https://github.com/anonymoussubmissio/CostCert/blob/main/drs%2Bpc_readcsv.py) to quickly get and analyze the saved results from step 3.   Note that to run [drs+pc_readcsv.py](https://github.com/anonymoussubmissio/CostCert/blob/main/drs%2Bpc_readcsv.py), you may conduct a similar procedure like step1-3 to corresponding ones (mark with (for extended CostCert)) for masking-based base model.

