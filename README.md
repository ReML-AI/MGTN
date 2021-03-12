# Modular Graph Transformer Networks (MGTN)
This project implements the multi-learning based on Modular Graph Transformer Networks (MGTN). 

### Requirements
Please, install the following packages
- numpy
- pytorch (1.*)
- torchnet
- torchvision
- tqdm
- networkx

### Download best checkpoints
checkpoint/coco/mgtn_final_86.9762.pth.tar ([Dropbox](https://www.dropbox.com/s/fr2286gwxsg80kq/mgtn_final_86.9762.pth.tar?dl=0))

### Performance

| Method                  | mAP       | CP       | CR         | CF1       | OP       | OR        | OF1       |
| ----------------------- | --------- | -------- | ---------- | --------- | -------- | --------- | --------- |
| CNN-RNN                 | 61.2      | -        | -          | -         | -        | -         | -         |
| SRN                     | 77.1      | 81.6     | 65.4       | 71.2      | 82.7     | 69.9      | 75.8      |
| Baseline(ResNet101)     | 77.3      | 80.2     | 66.7       | 72.8      | 83.9     | 70.8      | 76.8      |
| Multi-Evidence          | –         | 80.4     | 70.2       | 74.9      | 85.2     | 72.5      | 78.4      |
| ML-GCN (2019)           | 82.4      | 84.4     | 71.4       | 77.4      | 85.8     | 74.5      | 79.8      |
| ML-GCN (ResNeXt50 swsl) | 86.2      | 85.8     | 77.3       | 81.3      | 86.2     | 79.7      | 82.8      |
| A-GCN                   | 83.1      | 84.7     | 72.3       | 78.0      | 85.6     | 75.5      | 80.3      |
| KSSNet                  | 83.7      | 84.6     | 73.2       | 77.2      | 87.8     | 76.2      | 81.5      |
| SGTN (Our**)            | 86.6      | 77.2     | **82.2**   | 79.6      | 76.0     | **82.6**  | 79.2      |
| **MGTN(Base)**          | 86.9      | **89.4** | 74.5       | 81.3      | **90.9** | 76.3      | 83.0      |
| **MGTN(Final}**         | **87.0**  | 86.1     | 77.9       | **81.8**  | 87.7     | 79.4      | **83.4**  |

** SGTN (Our): https://github.com/ReML-AI/sgtn 

### TGCN on COCO

```sh
python main.py data/coco --image-size 448 --workers 8 --batch-size 32 --lr 0.03 --learning-rate-decay 0.1 --epoch_step 20 30 --embedding model/embedding/coco_glove_word2vec_80x300_ec.pkl --adj-strong-threshold 0.4 --adj-weak-threshold 0.2 --device_ids 0 1 2 3
```

### How to cite this work?
```
@inproceedings{Nguyen:AAAI:2021,
	author = {Nguyen, Hoang D. and Vu, Xuan-Son and Le, Duc-Trong},
	title = {Modular Graph Transformer Networks for Multi-Label Image Classification},
	booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
	series = {AAAI '21},
	year = {2021},
	publisher = {AAAI}
}
```



## Reference
This project is based on the following implementations:

- https://github.com/ReML-AI/sgtn
- https://github.com/durandtibo/wildcat.pytorch
- https://github.com/tkipf/pygcn
- https://github.com/Megvii-Nanjing/ML_GCN/
- https://github.com/seongjunyun/Graph_Transformer_Networks


