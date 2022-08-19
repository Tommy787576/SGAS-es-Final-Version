# SGAS-es: Avoiding Performance Collapse by Sequential Greedy Architecture Search with the Early Stopping Indicator

### Overview
Sequential Greedy Architecture Search (SGAS) reduces the discretization loss of Differentiable Architecture Search (DARTS). However, we observed that SGAS may lead to unstable searched results as DARTS. We referred to this problem as the cascade performance collapse issue. Therefore, we proposed Sequential Greedy Architecture Search with the Early Stopping Indicator (SGAS-es). We adopted the early stopping mechanism in each phase of SGAS to stabilize searched results and further improve the searching ability. The early stopping mechanism is based on the relation among Flat Minima, the largest eigenvalue of the Hessian matrix of the loss function, and performance collapse. We devised a mathematical derivation to show the relation between Flat Minima and the largest eigenvalue. The moving averaged largest eigenvalue is used as an early stopping indicator. Finally, we used NAS-Bench-201 and Fashion-MNIST to confirm the performance and stability of SGAS-es. Moreover, we used EMNIST-Balanced to verify the transferability of searched results. These experiments show that SGAS-es is a robust method and can derive the architecture with good performance and transferability.

### Codebase
This code is based on the [SGAS implementation](https://github.com/lightaime/sgas) and [RobustDARTS implementation](https://github.com/automl/RobustDARTS).

### Please switch to ./cnn directory first
``` 
cd cnn
```

### Search on FashionMNIST
``` 
python3 train_search.py --dataset=fashionMNIST
```

### Retrain on FashionMNIST
<arch_name> is the searched cell recorded in genotypes.py, such as `FashionMNIST_1`, `FashionMNIST_2`, and `FashionMNIST_3`.
``` 
python3 train.py --auxiliary --cutout --rand_erase --arch=<arch_name> --dataset=fashionMNIST
```
If you want to load check point while retraining:
```
python3 train.py --epoch=<remain epochs> --learning_rate=<current lr>  --load_check --load_path=<path of weight.pt> --auxiliary --cutout --rand_erase --arch=<arch_name> --dataset=fashionMNIST
```

### Retrain on EMNIST-Balanced
```
python3 train.py --epoch=200 --batch_size=96 --auxiliary --arch=<arch_name> --dataset=EMNIST-balanced
```

### Pretrained models
To test pretrained models in ./pretrained_models directory (take FashionMNIST_1.pt as an example):
``` 
python test.py --auxiliary --arch=FashionMNIST_1 --model_path=<path of FashionMNIST_1.pt>
```