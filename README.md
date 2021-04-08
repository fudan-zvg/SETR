# SEgmentation TRansformers -- SETR

![image](fig/image.png)

> [**Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers**](https://arxiv.org/abs/2012.15840),            
> Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, Li Zhang,        
> *CVPR 2021* 


## Installation

Our project is developed based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the official mmsegmentation [INSTALL.md](docs/install.md) and [getting_started.md](docs/getting_started.md) for installation and dataset preparation.

### A from-scratch setup script

#### Linux

Here is a full script for setting up SETR with conda and link the dataset path (supposing that your dataset path is $DATA_ROOT).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
git clone https://github.com/fudan-zvg/SETR.git
cd SETR
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt

mkdir data
ln -s $DATA_ROOT data
```

#### Windows(Experimental)

Here is a full script for setting up SETR with conda and link the dataset path (supposing that your dataset path is
%DATA_ROOT%. Notice: It must be an absolute path).

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
set PATH=full\path\to\your\cpp\compiler;%PATH%
pip install mmcv

git clone https://github.com/fudan-zvg/SETR.git
cd SETR
pip install -e .  # or "python setup.py develop"
pip install -r requirements/optional.txt

mklink /D data %DATA_ROOT%
```

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

### Pre-trained model
When you run the training command, the pre-trained model will be automatically downloaded and placed in a suitable location. If you are unable to download due to network reasons, you can download the pre-trained model by clicking [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth) and [here](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth) (DeiT), and then put it in `/root/.cache/torch/hub/checkpoints/`(this is the path on my computer, you can get the correct one from the log of the training command).

### Train

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
# For example, train a SETR-PUP on Cityscapes dataset with 8 GPUs
./tools/dist_train.sh configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8.py 8
```

* Tensorboard
If you want to use Tensorboard, install tensorboard and SETR/configs/_base_/default_runtime.py

If you want to use tensorboard, you need to `pip install tensorflow` and uncomment the Line 6 `dict(type='TensorboardLoggerHook')` of `SETR/configs/_base_/default_runtime.py`.


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

### Generate the png files to be submit to the official evaluation server

* Cityscapes

    First, add following to config file `configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8.py`,

    ```python
    data = dict(
        test=dict(
            img_dir='leftImg8bit/test',
            ann_dir='gtFine/test'))
    ```

   Then run test.

    ```shell
    ./tools/dist_test.sh configs/SETR/SETR_PUP_768x768_40k_cityscapes_bs_8.py \
        work_dirs/SETR_PUP_768x768_40k_cityscapes_bs_8/iter_40000.pth \
        8 --format-only --eval-options "imgfile_prefix=./SETR_PUP_768x768_40k_cityscapes_bs_8_test_results"
    ```

    You will get png files under `./SETR_PUP_768x768_40k_cityscapes_bs_8_test_results` directory.
    You may run `zip -r SETR_PUP_768x768_40k_cityscapes_bs_8_test_results.zip SETR_PUP_768x768_40k_cityscapes_bs_8_test_results/` and submit the zip file to [evaluation server](https://www.cityscapes-dataset.com/submit/).

* ADE20k

    ADE20k dataset could be download from this [link](http://sceneparsing.csail.mit.edu/)

    First, add following to config file `configs/SETR/SETR_PUP_512x512_160k_ade20k_bs_16.py`,

    ```python
    data = dict(
        test=dict(
            img_dir='images/testing',
            ann_dir='annotations/testing'))
    ```

   Then run test.

    ```shell
    ./tools/dist_test.sh configs/SETR/SETR_PUP_512x512_160k_ade20k_bs_16.py \
        work_dirs/SETR_PUP_512x512_160k_ade20k_bs_16/iter_1600000.pth \
        8 --format-only --eval-options "imgfile_prefix=./SETR_PUP_512x512_160k_ade20k_bs_16_test_results"
    ```

    You will get png files under `./SETR_PUP_512x512_160k_ade20k_bs_16_test_results` directory.
    You may run `zip -r SETR_PUP_512x512_160k_ade20k_bs_16_test_results.zip SETR_PUP_512x512_160k_ade20k_bs_16_test_results/` and submit the zip file to [evaluation server](http://sceneparsing.csail.mit.edu/eval/login.php).


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

