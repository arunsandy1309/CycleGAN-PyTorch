# CycleGAN - PyTorch Implementation

This repository contains the Pytorch implementation of the following paper:
>**Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**</br>
>Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros</br>
>https://arxiv.org/abs/1703.10593
>
>**Abstract:** _Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G : X → Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F : Y → X and introduce a cycle consistency loss to enforce F(G(X)) ≈ X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach._

## Architecture
<img src="https://user-images.githubusercontent.com/50144683/233066107-57c019b0-6357-4a2d-849a-a04f6b355971.png"></br>
(a) Our model contains two mapping functions G : X → Y and F : Y → X, and associated adversarial discriminators DY and DX. DY encourages G to translate X into outputs indistinguishable from domain Y , and vice versa for DX and F. To further regularize the mappings, we introduce two cycle consistency losses that capture the intuition that if we translate from one domain to the other and back again we should arrive at where we started:</br> 
(b) forward cycle-consistency loss: x → G(x) → F(G(x)) ≈ x, and </br>
(c) backward cycle-consistency loss: y → F(y) → G(F(y)) ≈ y </br>
