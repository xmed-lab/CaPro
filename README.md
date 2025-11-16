# **Capro**

## Introduction

Curvilinear Structure Segmentation (CSS) is crucial for applications like medical imaging and structural health monitoring. While the Segment Anything Model (SAM) shows potential for CSS, its direct application yields poor results, and existing adaptation methods rely heavily on costly pixel-level annotations and numerous training samples. This paper addresses a more challenging and practical scenario: adapting SAM using only a *single unlabeled image*. To this end, we propose **Curvilinear-aware Prompt Learning (CaPro)**, a fine-tuning-free framework. *CaPro* operates in two stages: first, it synthesizes curvilinear structures and trains a self-supervised *oriented* object detector to generate prompts; second, it introduces a **curvilinear-aware discrete representation matching** mechanism to filter out unreliable prompts by leveraging shared topological patterns with handwritten digits. This approach enables cost-effective and annotation-free adaptation of SAM to CSS tasks, demonstrating significant performance improvements.

![CaPro框架介绍图](intro.png)

Models may be needed: https://pan.baidu.com/s/1i7Kid7Io943dJNiH39GylQ?pwd=1fj5



### Co-first Author
**These authors contributed equally.**
- Zhuangzhuang Chen
- Qiangyu Chen

## **Installation**

#### **1. Clone the Repository:**

```bash
git clone https://github.com/QiangyuChen1/Capro.git
```

#### **2. Create a conda environment:**

```bash
conda create -n Capro python=3.11.5
conda activate Capro
```

#### **3. Install dependencies:**

```bash
cd Capro/
pip install -r requirements.txt
```

## **Data Generation**

The `generate_data.py` script in the scripts directory of our project encapsulates the complete data generation process. This process can be broadly divided into the following three steps.

1. Generate a curve structure. The parameters can be adjusted as required in `make_fakevessel.py`
2. Embed the curve structure into the image. The parameters can be adjusted as required in `FDA_retinal.py`. 
3. Convert data format. Specifically, it refers to the RoLabelIMG_Transform module.

> [!WARNING]
>
> Before generating the data, please place the background reference image that needs to be embedded in the ./data/L-System/FDA/Single_image directory.

#### **Generating data**

```bash
python .\scripts\generate_data.py --num[num] # If the value of num is too small, it may cause an error.
```

Note: By modifying **line 36 of `convert_gt.py`**, you can select the background color (black/white) for the desired generated curve structure.

### Usage of Detector

------

By the first step: **Data generation**, we can find corresponding under detector/data/annotations of the train. Besides,  you can choose your own **val. json**.  Then, in the images directory, we will add our training images (if you follow our instructions exactly, we will add the images from **detector\RoLabelImg_Transform\img**).

```bash
cd .\detector\
python .\train.py
```

[^train.py]: train.py is where you can adjust the training parameters you need, but we'll use resnet50 as an example in this tutorial.

We then use `predict.py` to generate our predictions, and place the images we want to predict in the imgs directory.

```bash
python .\predict.py
```

The output is easy to find. What we need is the data in **txt_result**, which corresponds to the detection boxes for those images.

### CaPro

------

We filter the boxes we need with `TopK_Box.py`, where **k** is a hyperparameter. **original_txt** and **imgs** in the capro\dataset directory are the txt predicted using the detector in the previous step and the corresponding **original image**

```bash
cd .\capro\
python .\TopK_Box.py
```

#### Optional configuration

The following are optional parameters:

- Model type for SAM
- Input and output paths

> [!CAUTION]
>
> In some cases, when you run `Topk_Box.py` with the original number of boxes **less than K**, you won't be able to generate the corresponding txt file. In that case, you can assume that the original txt file will be sufficient (if it's a minority). **A fallback txt** read directory is provided in `CaPro_SAM`.py to solve this problem.



```
python .\CaPro_SAM.py
```

Here is a simple evaluation function to evaluate the quality of the image segmentation. We use the F1 and MIoU metrics

```bash
python .\eval.py
```

## 🚀 Implementation reference and Express our gratitude

- https://github.com/facebookresearch/segment-anything
- https://github.com/TY-Shi/FreeCOS
- https://github.com/ZeroE04/R-CenterNet

## 😄 Feel free to contact us

- qiangyuchen516@gmail.com

## Citation

If you find our paper is helpful in your research or applications, please consider citing:

```

```




