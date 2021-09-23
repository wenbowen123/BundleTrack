# LF-Net: Learning Local Features from Images

This repository is a tensorflow implementation  for Y.  Ono, E. Trulls, P. Fua,
K.M. Yi, "LF-Net: Learning Local Features from Images". If you use this code in
your research, please cite [the paper](https://arxiv.org/abs/1805.09662). 


![comparison](/teasers/teasers.png)

## Important Note regarding the use of ratio tests

Do **NOT** use the ratio test for descriptor matching! The commonly-used ratio 
test depends on the distribution of descriptor distances, and the threshold 
differs from one descriptor to another. Commonly used thresholds (0.9 0.7) are
actually harmful for LF-Net. If you want to use the ratio test, you need to 
either tune this manually, or use statistical analysis as Lowe did for SIFT.

## Installation

This code is based on Python3.6.5 and tensorflow with CUDA-8.0. For more details on
the required  libraries, see  `requirements.txt`. You  can also  easily prepare
this by doing

```
pip install -r requirements.txt
```

## Docker image

We created a self-contained [Docker image](https://hub.docker.com/r/jiangweivcg/lf-net-release-env), for running the keypoint extraction demo easily. Make sure you have the nvidia docker runtime.

To launch a container:

`docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -ti --name lf-net-container -v /path/to/code_repo:/home jiangweivcg/lf-net-release-env`

To run the ` run_lfnet.py` script inside the container:

`cd /home`

`python run_lfnet.py --in_dir=/path/to/images --out_dir=/path/to/outputs`


## Pretrained models and example dataset

Download                             the                            [pretrained
models](http://webhome.cs.uvic.ca/~kyi/files/2018/lf-net/pretrained.tar.gz) and
the                                                                [scare_coeur
sequence](http://webhome.cs.uvic.ca/~kyi/files/2018/lf-net/sacre_coeur.tar.gz). Extract
them to the current folder so that they fall under `release/models/outdoor` for
example.

For other datasets, we do not plan to release them at the moment. Please do not
contact us for  explanations on the training phase. We  are providing them **as
is** as a reference implementation.

### Updates since the arXiv version

The provided pre-trained  models are trained with full  360 degree augmentation
for  orientation. Thus,  the results  you get  from these  models are  slightly
different  from  the  one  reported  in  arXiv.  We  have  further  included  a
consistency term on the orientation assignment.

## Running the keypoint extraction demo

To run LF-Net for all images in a given directory, simply type:

```
python run_lfnet.py --in_dir=images --out_dir=outputs
```

In addition, you can easily do the 2-view matching demo through
`notebooks/demo.ipynb` .

## Training

Training code can be found in `train_lfnet.py`. We will **not** provide any
support for the training process and datasets. All issues related to this topic
will be closed without answers.


## Some Examples

| Outdoor dataset</br> Top: LF-Net, Bottom: SIFT | Indoor dataset </br>Top: LF-Net, Bottom: SIFT | Webcam dataset</br>Top: LF-Net, Bottom: SIFT |
|:---------|:--------------------|:----------------|
| ![outdoor](/teasers/sfm_ours_sift.gif)     | ![indoor](/teasers/scannet_ours_sift.gif) | ![webcam](/teasers/webcam_ours_sift.gif) |
