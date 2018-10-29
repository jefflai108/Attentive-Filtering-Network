# Attentive Filtering Network
University of Edinbrugh-Johns Hopkins University's system for ASVspoof 2017 Version 2.0 dataset. Work is in submission for ICASSP 2019.\\
paper on ArXiv: TO DO 

## Author 
Cheng-I Lai, Alberto Abad, Korin Richmond, Junichi Yamagishi, Najim Dehak, Simon King 

## Abstract 
An attacker may use a variety of techniques to fool an automatic speaker verification system into accepting them as a genuine user. Anti-spoofing methods meanwhile aim to make the system robust against such attacks. The ASVspoof 2017 Challenge focused specifically on replay attacks, with the intention of measuring the limits of replay attack detection as well as developing countermeasures against them. In this work, we propose our replay attacks detection system - Attentive Filtering Network, which is composed of an attention-based filtering mechanism that enhances feature representations in both the frequency and time domains, and a ResNet-based classifier. We show that the network enables us to visualize the automatically acquired feature representations that are helpful for spoofing detection. Attentive Filtering Network attains an evaluation EER of 8.99% on the ASVspoof 2017 Version 2.0 dataset. With system fusion, our best system further obtains a 30% relative improvement over the ASVspoof 2017 enhanced baseline system.

## Instruction to run 
TO DO 

## Visualization of attention heatmaps with different non-linearities
original 
![alt text](github_image/original)
![original]("https://github.com/jefflai108/Attentive-Filtering-Network/raw/master/github_image/original.png")

sigmoid 
![alt text](https://github.com/jefflai108/Attentive-Filtering-Network.git/github_image/sigmoid)

softmaxF (softmax on feature dimension)
![alt text](https://github.com/jefflai108/Attentive-Filtering-Network.git/github_image/softmaxF)

softmaxT (softmax on time dimension)
![alt text](https://github.com/jefflai108/Attentive-Filtering-Network.git/github_image/softmaxT)
