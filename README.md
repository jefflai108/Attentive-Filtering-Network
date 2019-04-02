# Attentive Filtering Network
University of Edinbrugh-Johns Hopkins University's system for ASVspoof 2017 Version 2.0 dataset. Published in ICASSP 2019.

Read [our paper](https://arxiv.org/abs/1810.13048) for more details. 

If you find the code useful, please cite
```
@inproceedings{lai2018attentive,
  title={Attentive Filtering Networks for Audio Replay Attack Detection},
  author={Lai, Cheng-I and Abad, Alberto and Richmond, Korin and Yamagishi, Junichi and Dehak, Najim and King, Simon},
  booktitle={2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2019},
  organization={IEEE}
}
```

## Abstract 
An attacker may use a variety of techniques to fool an automatic speaker verification system into accepting them as a genuine user. Anti-spoofing methods meanwhile aim to make the system robust against such attacks. The ASVspoof 2017 Challenge focused specifically on replay attacks, with the intention of measuring the limits of replay attack detection as well as developing countermeasures against them. In this work, we propose our replay attacks detection system - Attentive Filtering Network, which is composed of an attention-based filtering mechanism that enhances feature representations in both the frequency and time domains, and a ResNet-based classifier. We show that the network enables us to visualize the automatically acquired feature representations that are helpful for spoofing detection. Attentive Filtering Network attains an evaluation EER of 8.99% on the ASVspoof 2017 Version 2.0 dataset. With system fusion, our best system further obtains a 30% relative improvement over the ASVspoof 2017 enhanced baseline system.

## Visualization of attention heatmaps with different non-linearities
sigmoid
![sigmoid](https://github.com/jefflai108/Attentive-Filtering-Network/raw/master/github_image/sigmoid.png)

tanh
![tanh](https://github.com/jefflai108/Attentive-Filtering-Network/raw/master/github_image/tanh.png)

softmaxF (softmax on feature dimension)
![softmaxF (softmax on feature dimension)](https://github.com/jefflai108/Attentive-Filtering-Network/raw/master/github_image/softmaxF.png)

softmaxT (softmax on time dimension)
![softmaxT (softmax on time dimension)](https://github.com/jefflai108/Attentive-Filtering-Network/raw/master/github_image/softmaxT.png)

## ASVspoof 2017 Challenge & Dataset
http://www.asvspoof.org/index2017.html

## Getting Started 
1. Prerequisites: PyTorch 0.4
2. run.sh contains examples of training the networks on GPU 
3. main.py contains detials of the training details and configurations
4. src/attention_neuro/simple_attention_network.py contains the implementation of Attentive Filtering Network.

## Authors 
Cheng-I Lai, [Alberto Abad](https://www.l2f.inesc-id.pt/w/Alberto_Abad_Gareta), Korin Richmond, [Junichi Yamagishi](https://nii-yamagishilab.github.io), Najim Dehak, [Simon King](http://homepages.inf.ed.ac.uk/simonk/)

## Contact 
Cheng-I Jeff Lai: jefflai108@gmail.com
