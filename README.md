# CycleGAN - PyTorch Implementation

<img src="https://user-images.githubusercontent.com/50144683/233770559-119c3f75-c4ae-4e46-9f97-6c2ba4d7ad27.gif" width='100%'>This is our ongoing PyTorch implementation for unpaired image-to-image translation.</br>

This repository contains the Pytorch implementation of the following paper:
>**Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**</br>
>Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros</br>
>https://arxiv.org/abs/1703.10593
>
>**Abstract:** _Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G : X → Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F : Y → X and introduce a cycle consistency loss to enforce F(G(X)) ≈ X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach._

## Architecture
<img src="https://user-images.githubusercontent.com/50144683/233066107-57c019b0-6357-4a2d-849a-a04f6b355971.png"></br>
**(a)** Our model contains two mapping functions G : X → Y and F : Y → X, and associated adversarial discriminators DY and DX. DY encourages G to translate X into outputs indistinguishable from domain Y , and vice versa for DX and F. To further regularize the mappings, we introduce two cycle consistency losses that capture the intuition that if we translate from one domain to the other and back again we should arrive at where we started:</br> 
**(b)** forward cycle-consistency loss: x → G(x) → F(G(x)) ≈ x, and </br>
**(c)** backward cycle-consistency loss: y → F(y) → G(F(y)) ≈ y </br>

## Installation
+ Install python libraries visdom and dominate.
```
pip install visdom
pip install dominate
```

## Datasets
+ Download the CycleGAN dataset from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/), extract the files and place the folder in the "datasets" folder. Below are the details about the datasets:
+ `facades`: 400 images from the CMP Facades dataset.
+ `cityscapes`: 2975 images from the Cityscapes training set.
+ `maps`: 1096 training images scraped from Google Maps.
+ `horse2zebra`: 939 horse images and 1177 zebra images downloaded from ImageNet using keywords wild horse and zebra
+ `apple2orange`: 996 apple images and 1020 orange images downloaded from ImageNet using keywords apple and navel orange.
+ `summer2winter_yosemite`: 1273 summer Yosemite images and 854 winter Yosemite images were downloaded using Flickr API. See more details in our paper.
+ `monet2photo`, `vangogh2photo`, `ukiyoe2photo`, `cezanne2photo`: The art images were downloaded from Wikiart. The real photos are downloaded from Flickr using the combination of the tags landscape and landscapephotography. The training set size of each class is Monet:1074, Cezanne:584, Van Gogh:401, Ukiyo-e:1433, Photographs:6853.
+ `iphone2dslr_flower`: both classes of images were downlaoded from Flickr. The training set size of each class is iPhone:1813, DSLR:3316. See more details in our paper.

To train a model on your own datasets, you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain A and B. You can test your model on your training set by setting `phase='train'` in test.lua. You can also create subdirectories `testA` and `testB` if you have test data.
![Paper](https://user-images.githubusercontent.com/50144683/233865440-f73d3ef5-b3ff-48df-9623-427dad0ec623.jpg)

## Training/Test Details
+ See `options/train_options.py` and `options/base_options.py` for training flags; see `options/test_options.py` and `options/base_options.py` for test flags.
+ CPU/GPU (default `--gpu_ids 0`): Set `--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g. `--batchSize 32`) to benefit from multiple gpus.
+ During training, the current results can be viewed using two methods. First, if you set `--display_id` > 0, the results and loss plot will be shown on a local graphics web server launched by visdom. To do this, you should have visdom installed and a server running by the command `python -m visdom.server`. The default server URL is `http://localhost:8097`. `display_id` corresponds to the window ID that is displayed on the visdom server. The `visdom` display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id 0`. Second, the intermediate results are saved to `[opt.checkpoints_dir]/[opt.name]/web/` as an HTML file. To avoid this, set `--no_html`.

## Train/Test a Model
+ Train a Model:
```
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan
```
+ To view training results, check out `./checkpoints/horse2zebra_cyclegan/web/index.html`
+ Test the Model:
```
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan --phase test
```
+ The test results will be saved to a html file here: `/results/horse2zebra_cyclegan/test_latest/index.html`

## Related Works
+ [Generating Real World Images using DCGAN in PyTorch](https://github.com/arunsandy1309/RealWorld-Image-Generation-DCGAN)
+ [Anime Character Generation using DCGAN in PyTorch](https://github.com/arunsandy1309/Anime-Character-Generation-DCGAN)
+ [Conditional GAN in PyTorch](https://github.com/arunsandy1309/Conditional-GAN-PyTorch)
+ [Vanilla GAN in PyTorch](https://github.com/arunsandy1309/Vanilla-GAN)
