# SEgmentation TRansformers -- SETR

![image](fig/image.png)

[Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers](https://arxiv.org/abs/2012.15840)

CVPR 2021


## Installation

Our project is developed based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the official mmsegmentation [INSTALL.md](docs/install.md) for installation and dataset preparation.




## Main results


#### Cityscapes

| Method     | Crop Size | Batch size | iteration | set  | mIoU  |                          | 
| ---------- | --------- | ---------- | --------- | ---- | ----- | -----------------------  |
| SETR-Naive | 768x768   | 8          | 40k       | val  |       |  [model]() [config]()     | 
| SETR-Naive | 768x768   | 8          | 80k       | val  |       |  [model]() [config]()     | 
| SETR-MLA   | 768x768   | 8          | 40k       | val  |       |  [model]() [config]()     | 
| SETR-MLA   | 768x768   | 8          | 80k       | val  |       |  [model]() [config]()     | 
| SETR-PUP   | 768x768   | 8          | 40k       | val  |       |  [model]() [config]()     |
| SETR-PUP   | 768x768   | 8          | 80k       | val  |       |  [model]() [config]()     |
| SETR-Naive-DeiT | 768x768   | 8          | 40k       | val  |       |  [model]() [config]()     | 
| SETR-Naive-DeiT | 768x768   | 8          | 80k       | val  |       |  [model]() [config]()     | 
| SETR-MLA-DeiT   | 768x768   | 8          | 40k       | val  |       |  [model]() [config]()     | 
| SETR-MLA-DeiT   | 768x768   | 8          | 80k       | val  |       |  [model]() [config]()     | 
| SETR-PUP-DeiT   | 768x768   | 8          | 40k       | val  |       |  [model]() [config]()     |
| SETR-PUP-DeiT   | 768x768   | 8          | 80k       | val  |       |  [model]() [config]()     |

#### ADE20K

| Method   | Crop Size | Batch size | iteration | set  | mIoU  |                           | 
| -------- | --------- | ---------- | --------- | ---- | ----- | -----------------------   |
| SETR-MLA | 512x512   | 8          | 160k      | val  |       |   [model]()[config]()     | 
| SETR-MLA | 512x512   | 16         | 160k      | val  |       |   [model]()[config]()     | 
| SETR-PUP | 512x512   | 16         | 160k      | val  |       |   [model]()[config]()     | 

#### Pascal Context

| Method   | Crop Size | Batch size | iteration | set  | mIoU  |                           | 
| -------- | --------- | ---------- | --------- | ---- | ----- | -----------------------   |
| SETR-MLA | 480x480   | 8          | 80k       | val  |       |   [model]()[config]()     | 
| SETR-MLA | 480x480   | 16         | 80k       | val  |       |   [model]()[config]()     | 
| SETR-PUP | 480x480   | 16         | 80k       | val  |       |   [model]()[config]()     | 





## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMSegmentation.



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


## Acknowledgement

Thanks to previous open-sourced repo:  
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)     
[pytorch-image-models](https://github.com/rwightman/pytorch-image-models)  

