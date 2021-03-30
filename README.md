# SEgmentation TRansformers -- SETR

![image](fig/image.png)

> [**Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers**](https://arxiv.org/abs/2012.15840),            
> Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, Li Zhang,        
> *CVPR 2021* 


## Installation

Our project is developed based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the official mmsegmentation [INSTALL.md](docs/install.md) and [getting_started.md](docs/getting_started.md) for installation and dataset preparation.

## Main results


#### Cityscapes

| Method     | Crop Size | Batch size | iteration | set  | mIoU  |                          |
| ---------- | --------- | ---------- | --------- | ---- | ----- | -----------------------  |
| SETR-Naive | 768x768   | 8          | 40k       | val  | 77.37 | [model]() [config](configs/SETR/SETR_Naive_768x768_40k_cityscapes_bs_8.py) |
| SETR-Naive | 768x768   | 8          | 80k       | val  | 77.90 |  [model]() [config](configs/SETR/SETR_Naive_768x768_80k_cityscapes_bs_8.py)     |
| SETR-MLA   | 768x768   | 8          | 40k       | val  | 76.65 |  [model]() [config](configs/SETR/SETR_MLA_768x768_40k_cityscapes_bs_8.py)     |
| SETR-MLA   | 768x768   | 8          | 80k       | val  | 77.24 |  [model]() [config](configs/SETR/SETR_MLA_768x768_80k_cityscapes_bs_8.py)     |
| SETR-PUP   | 768x768   | 8          | 40k       | val  | 78.39 |  [model]() [config](configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8.py)     |
| SETR-PUP   | 768x768   | 8          | 80k       | val  | 79.34 |  [model]() [config](configs/SETR/SETR_PUP_768x768_80k_cityscapes_bs_8.py)     |
| SETR-Naive-DeiT | 768x768   | 8          | 40k       | val  | 77.85 |  [model]() [config](configs/SETR/SETR_Naive_DeiT_768x768_40k_cityscapes_bs_8.py)     |
| SETR-Naive-DeiT | 768x768   | 8          | 80k       | val  | 78.66 |  [model]() [config](configs/SETR/SETR_Naive_DeiT_768x768_80k_cityscapes_bs_8.py)     |
| SETR-MLA-DeiT   | 768x768   | 8          | 40k       | val  | 78.04 |  [model]() [config](configs/SETR/SETR_MLA_DeiT_768x768_40k_cityscapes_bs_8.py)     |
| SETR-MLA-DeiT   | 768x768   | 8          | 80k       | val  | 78.98 |  [model]() [config](configs/SETR/SETR_MLA_DeiT_768x768_80k_cityscapes_bs_8.py)     |
| SETR-PUP-DeiT   | 768x768   | 8          | 40k       | val  | 78.79 |  [model]() [config](configs/SETR/SETR_PUP_DeiT_768x768_40k_cityscapes_bs_8.py)     |
| SETR-PUP-DeiT   | 768x768   | 8          | 80k       | val  | 79.45 |  [model]() [config](configs/SETR/SETR_PUP_DeiT_768x768_80k_cityscapes_bs_8.py)     |

#### ADE20K

| Method     | Crop Size | Batch size | iteration | set  | mIoU  | mIoU(ms+flip) |                                                              |
| ---------- | --------- | ---------- | --------- | ---- | ----- | ------------- | ------------------------------------------------------------ |
| SETR-Naive | 512x512   | 16         | 160k      | Val  | 48.06 | 48.80         | [model]() [config](configs/SETR/SETR_Naive_512x512_160k_ade20k_bs_16.py) |
| SETR-MLA   | 512x512   | 8          | 160k      | val  | 48.27 | 50.03         | [model]() [config](configs/SETR/SETR_MLA_512x512_160k_ade20k_bs_8.py) |
| SETR-MLA   | 512x512   | 16         | 160k      | val  | 48.64 | 50.28         | [model]() [config](configs/SETR/SETR_MLA_512x512_160k_ade20k_bs_16.py) |
| SETR-PUP   | 512x512   | 16         | 160k      | val  | 48.58 | 50.09         | [model]() [config](configs/SETR/SETR_PUP_512x512_160k_ade20k_bs_16.py) |

#### Pascal Context

| Method     | Crop Size | Batch size | iteration | set  | mIoU  | mIoU(ms+flip) |                                                              |
| ---------- | --------- | ---------- | --------- | ---- | ----- | ------------- | ------------------------------------------------------------ |
| SETR-Naive | 480x480   | 16         | 80k       | val  | 52.89 | 53.61         | [model]() [config](configs/SETR/SETR_Naive_480x480_80k_pascal_context_bs_16.py) |
| SETR-MLA   | 480x480   | 8          | 80k       | val  | 54.39 | 55.39         | [model]() [config](configs/SETR/SETR_MLA_480x480_80k_pascal_context_bs_8.py) |
| SETR-MLA   | 480x480   | 16         | 80k       | val  | 54.87 | 55.83         | [model]() [config](configs/SETR/SETR_MLA_480x480_80k_pascal_context_bs_16.py) |
| SETR-PUP   | 480x480   | 16         | 80k       | val  | 54.40 | 55.27         | [model]() [config](configs/SETR/SETR_PUP_480x480_80k_pascal_context_bs_16.py) |


## Get Started


### Train

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train a SETR-PUP on Cityscapes dataset with 8 GPUs
./tools/dist_train.sh configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8.py 8
```

### Single-scale testing

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  [--eval ${EVAL_METRICS}]
# For example, test a SETR-PUP on Cityscapes dataset with 8 GPUs
./tools/dist_test.sh configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8.py \
work_dirs/SETR_PUP_768x768_40k_cityscapes_bs_8/iter_40000.pth \
8 --eval mIoU
```

### Multi-scale testing

Use the config file ending in `_MS.py` in `configs/SETR`.

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  [--eval ${EVAL_METRICS}]
# For example, test a SETR-PUP on Cityscapes dataset with 8 GPUs
./tools/dist_test.sh configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8_MS.py \
work_dirs/SETR_PUP_768x768_40k_cityscapes_bs_8/iter_40000.pth \
8 --eval mIoU
```

Please see [getting_started.md](docs/getting_started.md) for the more basic usage of training and testing.

## Reference

```bibtex
@inproceedings{SETR,
    title={Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers}, 
    author={Zheng, Sixiao and Lu, Jiachen and Zhao, Hengshuang and Zhu, Xiatian and Luo, Zekun and Wang, Yabiao and Fu, Yanwei and Feng, Jianfeng and Xiang, Tao and Torr, Philip H.S. and Zhang, Li},
    booktitle={CVPR},
    year={2021}
}
```

## License

MIT


## Acknowledgement

Thanks to previous open-sourced repo:  
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)     
[pytorch-image-models](https://github.com/rwightman/pytorch-image-models)  

