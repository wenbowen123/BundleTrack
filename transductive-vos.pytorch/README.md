## A Transductive Approach for Video Object Segmentation

<img src="figure/fig1.png" width = 60% height = 60%/>

This repo contains the pytorch implementation for the CVPR 2020 paper [A Transductive Approach for Video Object Segmentation](https://arxiv.org/abs/2004.07193).

## Pretrained Models and Results

We provide three pretrained models of ResNet50. They are trained from DAVIS 17 training set, combined DAVIS 17 training and validation set and YouTube-VOS training set.

- [Davis-train](https://drive.google.com/open?id=1SWZ20zTHgOpha0MlF8iOdqEHFkALdZn7)
- [Davis-trainval](https://drive.google.com/open?id=14Qm8UEQG-rYYepDYzKPTc1KqISzQkT95)
- [Youtube-train](https://drive.google.com/open?id=1U6sX9EUpOvDRFyaqDVpsi3plTnI2Witp)

Our pre-computed results can be downloaded [here](https://drive.google.com/open?id=1QdKaeoMU7KaEp0TIXZZNLdgm9n8IOQOj).

Our results on DAVIS17 and YouTube-VOS:

| Dataset              | J    | F    |
| -------------------- | :--- | ---- |
| DAVIS17 validation   | 69.9 | 74.7 |
| DAVIS17 test-dev     | 58.8 | 67.4 |
| YouTube-VOS (seen)   | 67.1 | 69.4 |
| YouTube-VOS (unseen) | 63.0 | 71.6 |

## Usage

- Install python3, pytorch >= 0.4, and PIL package.

- Clone this repo:

  ```shell
  git clone https://github.com/microsoft/transductive-vos.pytorch
  ```

- Prepare DAVIS 17 train-val dataset:

  ```shell
  # first download the dataset
  cd /path-to-data-directory/
  wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
  # unzip
  unzip DAVIS-2017-trainval-480p.zip
  # split train-val dataset
  python /VOS-Baseline/dataset/split_trainval.py -i ./DAVIS
  # clean up
  rm -rf ./DAVIS
  ```

  Now, your data directory should be structured like this:

  ```
  .
  |-- DAVIS_train
      |-- JPEGImages/480p/
          |-- bear
          |-- ...
      |-- Annotations/480p/
  |-- DAVIS_val
      |-- JPEGImages/480p/
          |-- bike-packing
          |-- ...
      |-- Annotations/480p/ 
  ```

- Training on DAVIS training set:

  ```shell
  python -m torch.distributed.launch --master_port 12347 --nproc_per_node=4 main.py --data /path-to-your-davis-directory/
  ```

  All the training parameters are set to our best setting to reproduce the ResNet50 model as default.  In this setting you need to have 4 GPUs with 16 GB CUDA memory each. Feel free to contact the author on parameter settings if you want to train on a single or more GPUs.

  If you want to change some parameters, you can see comments in `main.py` or

  ```shell
  python main.py -h
  ```

- Inference on DAVIS validation set, 1 GPU with 12 GB CUDA memory is needed:

  ```shell
  python inference.py -r /path-to-pretrained-model -s /path-to-save-predictions
  ```

  Same as above, all the inference parameters are set to our best setting on DAVIS validation set as default, which is able to reproduce our result with a J-mean of 0.699. The saved predictions can be directly evaluated by [DAVIS evaluation code](https://github.com/davisvideochallenge/davis2017-evaluation).
  
## Further Improvements
This approach is simple with clean implementations, if you add few tiny tricks, the performance will be furhter improved. For exmaple,
- If performing epoch test, i.e., selecting the best-performing epoch, you can further get ~1.5 points absolute performance improvements on DAVIS17 dataset.
- Pretraining the model on other image datasets with mask annotation, such as semantic segmentation and salient object detection, may bring further improvements. 
- ... ...

## Contact

For any questions, please feel free to reach

```
Yizhuo Zhang: criszhang004@gmail.com
Zhirong Wu: xavibrowu@gmail.com
```

## Citations
```
@inproceedings{zhang2020a,
  title={A Transductive Approach for Video Object Segmentation}
  author={Zhang, Yizhuo and Wu, Zhirong and Peng, Houwen and Lin, Stephen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
