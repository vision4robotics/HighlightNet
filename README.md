# HighlightNet: Highlighting Low-Light Potential Features for Real-Time UAV Tracking

This project inclueds code and demo videos of HighlightNet.

# Abstract 
>Low-light environments have posed a formidable challenge for robust UAV tracking even with state-of-the-art trackers since the potential image features are hard to extract under adverse light conditions. Besides, due to the low visibility, accurate online selection of the object also becomes extremely difficult for human monitors to initialize UAV tracking in ground control stations (GCSs). To address these problems, this work proposed a novel enhancer, i.e., HighlightNet, to light up potential objects for both human operators and UAV trackers. By employing Transformer, HighlightNet can adjust enhancement parameters according to global features and is thus adaptive for illumination variation. Pixel-level range mask is introduced to make HighlightNet more focused on the enhancement of the tracking object and regions without light sources. Furthermore, a soft truncation mechanism is built to prevent background noise from being mistaken for crucial features. Experiments on image enhancement benchmarks demonstrate HighlightNet has advantages in facilitating human perception. Evaluations on the public UAVDark135 benchmark show that HightlightNet is more suitable for UAV tracking tasks than other top-ranked low-light enhancers. In addition, with real-world tests on a typical UAV platform, HighlightNet verifies its practicability and efficiency in nighttime aerial tracking-related applications.
# Demo video


# Contact 
Haolin Dong

Email: 1851146@tongji.edu.cn

Changhong Fu

Email: changhongfu@tongji.edu.cn

# Demonstration running instructions

### Requirements

1.Python 3.7.10

2.Pytorch 1.10.1

4.torchvision 0.11.2

5.cuda 11.3.1

>Download the package, extract it and follow two steps:
>
>1. Put test images in data/test_data/, put training data in data/train_data/.
>
>2. For testing, run:
>
>     ```
>     python lowlight_test.py
>     ```
>     You can find the enhanced images in data/result/. Some examples have been put in this folder.
>   
>3. For training, run:
>
>     ```
>     python lowlight_train.py
>     ```



# Acknowledgements

We sincerely thank the contribution of `Chongyi Li` for his previous work Zero-DCE (https://github.com/Li-Chongyi/Zero-DCE).
