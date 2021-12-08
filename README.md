# DeepFM_pytorch

A PyTorch implementation of DeepFM.

# Usage

1）、Create env environment

```
bash deploy.sh
```

2）、Run training（Use multiple GPUs）

```
bash train.sh
```

after three epoch，the test auc（sample auc） and uid auc（user group auc） is:

2021-12-08 17:21:14,946 : INFO : test total auc is: 0.7427503575538603, test uid auc is: 0.7039701989774394
2021-12-08 17:21:15,213 : INFO : test total auc is: 0.7382420528977591, test uid auc is: 0.7129954531228962
2021-12-08 17:21:15,498 : INFO : test total auc is: 0.746973366258344, test uid auc is: 0.7165268359801436
2021-12-08 17:21:15,519 : INFO : test total auc is: 0.7431638676244506, test uid auc is: 0.7111437456524067
