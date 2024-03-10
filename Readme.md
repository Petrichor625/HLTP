#  HLTP: A Cognitive-Based Trajectory Prediction Approach for Autonomous Driving

## Overview

This repository contains the official implementation of **A Cognitive-Based Trajectory Prediction Approach for Autonomous Driving**, accepted by the journal **IEEE Transactions on Intelligent Vehicles.**

 ![image](https://github.com/Petrichor625/HLTP/blob/main/head_image.jpg)

## Highlights

- Incorporating a novel teacher-student knowledge distillation framework by employing a sophisticated multi-task learning strategy for knowledge distillation, effectively balancing and weighing multiple loss functions while considering the homoscedastic uncertainty associated
  with each task. 
- Introducing an advanced adaptive visual sector, a key innovation in the realm of vision-aware
  pooling mechanisms to mimic human visual procession.
- Proposing a novel attention block called the shift-window attention block to simulate the observational processes characteristic of human drivers, effectively capturing the nuances of human visual attention and spatial awareness in a computational format. 
- Exhibiting remarkable performance even with fewer input observations and in scenarios characterized by missing data, which demonstrates the model’s robustness and accuracy across various traffic conditions, including on highways and in dense urban environments.



## Abstract

In the field of autonomous vehicles (AVs), accurate trajectory prediction is essential for safe and efficient navigation. More importantly, the predicted trajectories must be consistent with human driving behavior, which is even safer and able to respond effectively to unexpected situations. Therefore, we present a Human-Like Trajectory Prediction (HLTP) model that emulates human cognitive processes for improved trajectory prediction in AVs. The HLTP model incorporates a sophisticated teacher-student knowledge distillation framework. The teacher model, equipped with an adaptive visual sector, mimics the visual processing of the human brain, particularly the functions of the occipital and temporal lobes. The student model focuses on real-time interaction and decision-making, drawing parallels to prefrontal and parietal cortex functions. This dual-model approach allows for dynamic adaptation to changing driving scenarios, capturing essential perceptual cues for accurate prediction. Evaluated using the Macao Connected and Autonomous Driving (MoCAD) dataset, along with the NGSIM and HighD benchmarks, HLTP demonstrates superior performance compared to existing models, particularly in challenging environments with incomplete data. The project page is available at https://github.com/Petrichor625/Map-Free-Behavior-Aware-Model.



## Framework

Overall “teacher-student” architecture of the HLTP. The Surround-aware encoder and the Teacher Encoder within the “teacher” model process visual vectors and context matrices to produce surround-aware and visual-aware vectors, respectively. These vectors are then fed into the Teacher Multimodal Decoder, which enables the prediction of different potential maneuvers for the target vehicle, each with associated probabilities. The “student” model acquires knowledge from the “teacher” model using a Knowledge Distillation Modulation (KDM) training strategy. This approach ensures accurate, human-like trajectory predictions even with minimal observational data.
![framework](https://github.com/Petrichor625/HLTP/blob/main/framework.png)




## Environment

- **Operating System**: Ubuntu 20.04
- **CUDA Version**: 11.4

 The NGSIM and HighD datasets in our work is segmented in the same way as the work ([stdan](https://ieeexplore.ieee.org/document/9767719))


## Train

To begin training your HLTP model with the NGSIM dataset, you can easily start with the provided script. Training is a crucial step to ensure your model accurately understands and processes the dataset.

First, you need to train the HLTP teacher model:

```
python train_teacher.py
```

Then, you need to train the HLTP student model by using specific pretrained teacher model:

```
python train_student.py
```

## Qualitative results

We are preparing a script for generating these visualizations:

 ````
 Code for qualitative results coming soon.
 ````

 ![image](https://github.com/Petrichor625/HLTP/blob/main/visual.gif)




## Evaluation

To evaluate the HLTP teacher model, you can use the following command to start the evaluation process:

```
python evaluate_teacher.py
```

To evaluate the HLTP student model, you can use the following command to start the evaluation process:

```
python evaluate_student.py
```

## Citation
**A Cognitive-Based Trajectory Prediction Approach for Autonomous Driving**, accepted by the journal **_IEEE Transactions on Intelligent Vehicles_.** (Camera-ready)

```

```
 


