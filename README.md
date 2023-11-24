# RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model 

* Zilun Zhang, Tiancheng Zhao, Yulong Guo, Jianwei Yin

* Preprint (Updated on Aug 31): https://arxiv.org/abs/2306.11300

* Data: https://huggingface.co/datasets/Zilun/RS5M/


##  RS5M Dataset

Pre-trained Vision-Language Foundation Models utilizing extensive image-text paired data have demonstrated unprecedented image-text association capabilities, achieving remarkable results across various downstream tasks. A critical challenge is how to make use of existing large-scale pre-trained VLMs, which are trained on common objects, to perform the domain-specific transfer for accomplishing domain-related downstream tasks. In this paper, we propose a new framework that includes the Domain Foundation Model (DFM), bridging the gap between the general foundation model (GFM) and domain-specific downstream tasks. Moreover, we present an image-text paired dataset in the field of remote sensing (RS), RS5M, which has 5 million remote sensing images with English descriptions. The dataset is obtained from filtering publicly available image-text paired datasets and captioning label-only RS datasets with pre-trained models. These constitute the first large-scale RS image-text paired dataset. Additionally, we tried several Parameter-Efficient Tuning methods with Vision-Language Models on RS5M as the baseline for the DFM. Experimental results show that our proposed datasets are highly effective for various tasks, improving upon the baseline by $\sim$ 16 % in zero-shot classification tasks, and obtaining good results in both Vision-Language Retrieval and Semantic Localization tasks.

![teaser](15datasets_teaser.png)

## GeoRSCLIP Model
### Installation

* Install Pytorch following instructions from the official website (We tested in torch 2.0.1 with CUDA 11.8 and 2.1.0 with CUDA 12.1)

```bash
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

* Install other dependencies

```bash
  pip install pillow pandas scikit-learn ftfy tqdm matplotlib transformers adapter-transformers open_clip_torch pycocotools timm clip-benchmark torch-rs
```

### Usage

* Clone the repo from: https://huggingface.co/Zilun/GeoRSCLIP

```bash
git clone https://huggingface.co/Zilun/GeoRSCLIP
cd GeoRSCLIP
```

* Unzip the test data
```bash
unzip data/rs5m_test_data.zip
```

* Run the inference script:
```bash
  python codebase/inference --ckpt-path /your/local/path/to/RS5M_ViT-B-32.pt --test-dataset-dir /your/local/path/to/rs5m_test_data
```

* (Optional) If you just want to load the GeoRSCLIP model:

```python

  import open_clip
  import torch
  from inference_tool import get_preprocess


  ckpt_path = "/your/local/path/to/RS5M_ViT-B-32.pt"
  model, _, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
  checkpoint = torch.load(ckpt_path, map_location="cpu")
  msg = model.load_state_dict(checkpoint, strict=False)
  model = model.to("cuda")
  img_preprocess = get_preprocess(
        image_resolution=224,
  )
```

```python

  import open_clip
  import torch
  from inference_tool import get_preprocess

  ckpt_path = "/your/local/path/to/RS5M_ViT-H-14.pt"
  model, _, _ = open_clip.create_model_and_transforms("ViT-H/14", pretrained="laion2b_s32b_b79k")
  checkpoint = torch.load(ckpt_path, map_location="cpu")
  msg = model.load_state_dict(checkpoint)
  model = model.to("cuda")
  img_preprocess = get_preprocess(
        image_resolution=224,
  )
```

### Experiment Result
||EuroSAT_acc|	RESISC45_acc|	AID_acc|	retrieval-image2text-R@1-rsitmd|	retrieval-image2text-R@5-rsitmd|	retrieval-image2text-R@10-rsitmd|	retrieval-text2image-R@1-rsitmd|	retrieval-text2image-R@5-rsitmd	|retrieval-text2image-R@10-rsitmd	|retrieval-mean-recall-rsitmd|	retrieval-image2text-R@1-rsicd|	retrieval-image2text-R@5-rsicd|	retrieval-image2text-R@10-rsicd|	retrieval-text2image-R@1-rsicd|	retrieval-text2image-R@5-rsicd|	retrieval-text2image-R@10-rsicd|	retrieval-mean-recall-rsicd	|Selo_Rsu|	Selo_Rda|	Selo_Ras|	Selo_Rmi|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|GeoRSCLIP-ViTB32|61.40|	72.74|	74.42|	17.92|	34.96|	46.02|	14.12|	41.46|	57.52|	35.33|	12.17|	28.45|	38.61|	9.31|	26.51|	41.28|	26.06|	0.755636|	0.730925|	0.258044|	0.744670|
|GeoRSCLIP-ViTH14|67.47|	73.83|	76.33|	23.45|	42.92|	53.32|	18.01|	44.60|	59.96|	40.38|	14.27|	29.55|	40.44|	11.38|	30.80|	44.41|	28.48|	0.759515|	0.741806|	0.256649|	0.749430|

## RS-SD
* We will retrain the RS-SD based on v5 version of RS5M dataset. The ckpt will be released here in 1 or 2 weeks.

## Dataset Download (About 500GB, 128 webdataset tars)
* Google Drive: 
  * https://drive.google.com/drive/folders/1NU6d9C50PUSzHdBy-wTi9HYLPHxtsHA3?usp=sharing
* Baidu Disk
  * https://pan.baidu.com/s/1wPYMN4lJRdHbYn4wT4HbHQ?pwd=recd
  * Password: recd
  * RS5M_v5 > data

* The BigEarthNet with RGB channels only (with corresponding filenames in our csv files)
   * https://pan.baidu.com/s/1aCqRmnCeow18ry__R_oZow?pwd=6ya9
   * Password: 6ya9

## MetaFile
* The metafile and other useful files of RS5M can be found here: https://huggingface.co/datasets/Zilun/RS5M/
* See README.md in huggingface for a breakdown explanation of each file.

## How to use this dataset

### Option 1 (Recommended)
* We create the webdataset format files containing paired image and text for sequential data io. Do **NOT** untar the files.

1. Download the webdataset files from the link provided above. The dataset directory should look like this:
   ```bash
       /nas/zilun/RS5M_v5/webdataset                                                       
       ├── train                        
           ├── pub11-train-0000.tar                                                         
           ├── pub11-train-0001.tar
           ├── ......
           ├── pub11-train-0030.tar                                         
           ├── pub11-train-0031.tar
           ├── rs3-train-0000.tar                                              
           ├── rs3-train-0001.tar
           ├── ......
           ├── rs3-train-0030.tar                                              
           ├── rs3-train-0031.tar
       ├── val                        
           ├── pub11-val-0000.tar                                                         
           ├── pub11-val-0001.tar
           ├── ......
           ├── pub11-val-0030.tar                                         
           ├── pub11-val-0031.tar
           ├── rs3-val-0000.tar                                              
           ├── rs3-val-0001.tar
           ├── ......
           ├── rs3-val-0030.tar                                              
           ├── rs3-val-0031.tar

    ```
3. An example of data IO pipeline using webdataset files is provided in "dataloader.py". The throughput (images per second) is ~1800 images per second. (With Ryzen 3950x CPU and dual-channel 3200MHZ DDR4 RAM)
4. Run the following to have a taste:
   ```bash
   python dataloader.py --train_dir /media/zilun/mx500/RS5M/data/train --val_dir /media/zilun/mx500/RS5M/data/val --num_worker 16 --batch_size 400 --num_shuffle 10000
   ```
### Option 2
* We also provide the pure image files, which could be used with the metafiles from huggingface. Due to the huge amount of the image data, an SSD drive is recommended.

1. Download the files from the [link](https://pan.baidu.com/s/1BIwkbA381diapxTsM4QGRA?pwd=ea86) provided. The dataset directory should look like this:
   ```bash
       /nas/zilun/RS5M_v5/img_only                                                      
       ├── pub11                        
           ├── pub11.tar.gz_aa                                                       
           ├── pub11.tar.gz_ab
           ├── ......
           ├── pub11.tar.gz_ba                                              
           ├── pub11.tar.gz_bc
       ├── rs3                        
           ├── ben
               ├── ben.tar.gz_aa                                       
           ├── fmow
               ├── fmow.tar.gz_aa
               ├── fmow.tar.gz_ab
               ├── ......
               ├── fmow.tar.gz_ap
               ├── fmow.tar.gz_aq
           ├── millionaid
               ├── millionaid.tar.gz_aa
               ├── millionaid.tar.gz_ab
               ├── ......
               ├── millionaid.tar.gz_ap
               ├── millionaid.tar.gz_aq                                   
    ```
2. Combine and untar the files. You will have the images files now.
    ```
     # optional, for split and zip the dataset
     tar -I pigz -cvf - BigEarthNet-S2-v1.0-rgb | split --bytes=500MB - ben.tar.gz_

     # combine different parts into one
     cat ben.tar* > ben.tar

     # extract
     tar -xvf ben.tar
    ```

## Statistics
### PUB11 Subset

| Name               | Amount |   After Keyword Filtering |   Download Image|  Invalid Image (Removed) |   Duplicate Image (Removed)|  Outlier images (Removed by VLM and RS Detector)|  Remain |
|:------------------:|:------:|:-------------------------:|:----------:|:------------------------:|:---------------------:|:------------------------------:|:--------:|
| LAION2B            | 2.3B   | 1,980,978   | 1,737,584   |             102          |        343,017        |          333,686               |1,060,779 |
| COYO700M           | 746M   | 680,089     | 566,076     |     28                   |245,650                |94,329                          | 226,069  |
| LAIONCOCO          | 662M   | 3,014,283   | 2,549,738   |       80                 |417,689                |527,941                         | 1,604,028|
| LAION400M          | 413M   | 286,102     | 241,324     |25                        |141,658                |23,860                          | 75,781    |
| WIT                | 37 M   | 98,540      | 93,754      |0                         |74,081                 |9,299                           | 10,374    |
| YFCC15M            | 15M    | 27,166      | 25,020      |0                         |265                    |15,126                          | 9,629     |
| CC12M              | 12M    | 18,892      | 16,234      | 0                        |1,870                  |4,330                           |10,034    |
| Redcaps            | 12M    | 2,842       | 2,686       | 0                        |228                    |972                             |1,486     |
| CC3M               | 3.3M   | 12,563      | 11,718      | 1                        |328                    |1,817                           |9,572     |
| SBU                | 1M     | 102         | 91          |0                         |4                      |36                              |51        |
| VG                 | 0.1M   | 26          | 26          | 0                        |0                      |20                              |6         |
| Total              | 4.2B   | 6,121,583   | 5,244,251   | 236                        |1,224,790              |1,011,416                       |3,007,809 |


### RS3 Subset

| Name               | Amount | Original Split | Has Class label |
|:------------------:|:------:|:--------------:|:---------------:|
|FMoW|727,144|Train|Yes|
|BigEarthNet|344,385|Train|Yes|
|MillionAID|990,848|Test|No|
|Total|2,062,377|-|-|

### Geo-Statistics
* Statistics of geometa for images contain the UTM zone, latitude, and longitude information.
  * YFCC14M: 7841
  * FMoW: 727,144
  * BigEarthNet: 344,385

  ![teaser](vis/geo_stats.png)

* Extract entity with "GPE" label using [NER from NLTK](https://medium.com/nirman-tech-blog/locationtagger-a-python-package-to-extract-locations-from-text-or-web-page-dbb05f1648d3)
  * Applied to captions in PUB11 subset
  * [Extraction Result](https://huggingface.co/datasets/Zilun/RS5M/blob/main/pub11_NER_geolocation_info.csv)
  * 880,354 image-text pairs contain "GPE", and most of them are city/country names.

## BLIP2 fine-tuned with RSITMD dataset
* Tuned with LoRA
* Checkpoint and inference code can be found through this [link](https://github.com/om-ai-lab/RS5M/tree/main/blip2_finetune)

## Image-Text Pair Rating Tool
* [Link](https://github.com/om-ai-lab/RS5M/tree/main/rating_app/rating_sys_webdataset)

## Awesome Remote Sensing Vision-Language Models & Papers

* https://github.com/om-ai-lab/awesome-RSVLM

## Contact
Email: zilun.zhang@zju.edu.cn

WeChat: zilun960822

Slack Group: https://join.slack.com/t/slack-nws5068/shared_invite/zt-1zpu3xt85-m8I3kVCp4qxAA1r1bDmKmQ

## Acknowledgement

We thank Delong Chen and his ITRA framework for helping us fine-tune the CLIP-like model.
https://itra.readthedocs.io/en/latest/Contents/introduction/overview.html


## BibTeX Citation

If you use RS5M in a research paper, we would appreciate using the following citations:

```
@misc{zhang2023rs5m,
      title={RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Foundation Model}, 
      author={Zilun Zhang and Tiancheng Zhao and Yulong Guo and Jianwei Yin},
      year={2023},
      eprint={2306.11300},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


