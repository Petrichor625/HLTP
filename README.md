🚗 **HLTP: Human-Like Trajectory Prediction**


# ⚠️ **Important Update**


🚀 **We’ve Uploaded the Latest Version!**  
We’re excited to announce that the repository now includes the **correct and complete version of the code**, along with all necessary model weights and components. This update should resolve any previous issues and allows you to fully replicate the results as described in our paper.

Please download or pull the latest version to ensure you have the most up-to-date files. Thank you for your patience and support!

---

🔗 **For any questions or support, feel free to reach out!**


## 📖 Overview

Welcome to the official repository for **A Cognitive-Based Trajectory Prediction Approach for Autonomous Driving**.

 ![image](https://github.com/Petrichor625/HLTP/blob/main/HLTP/HLTP/pic/visual.gif)
---


## ✨ Highlights

- **Knowledge Distillation Framework**  
  Employs a novel teacher-student framework utilizing a sophisticated multi-task learning strategy to balance multiple loss functions, taking into account homoscedastic uncertainty associated with each task.

- **Adaptive Visual Sector**  
  Introduces an advanced adaptive visual sector—a key innovation within vision-aware pooling mechanisms to closely mimic human visual processing.

- **Shift-Window Attention Block**  
  Features a new attention block designed to simulate human observational processes, effectively capturing the nuances of human visual attention and spatial awareness in a computational format.

- **Robust Performance**  
  Demonstrates outstanding robustness and accuracy with fewer input observations and even with missing data, performing consistently across diverse traffic conditions, including highways and dense urban environments.

---

## 📜 Abstract

Accurate trajectory prediction is vital for autonomous vehicles (AVs) to ensure safe and efficient navigation. To enhance safety and adaptability, predicted trajectories must align with human-like driving behavior. The **Human-Like Trajectory Prediction (HLTP)** model leverages a teacher-student knowledge distillation framework. The teacher model, equipped with an adaptive visual sector, mimics human brain visual processing (occipital and temporal lobes), while the student model focuses on real-time interaction and decision-making, reflecting functions of the prefrontal and parietal cortex. This dual-model approach dynamically adapts to evolving driving scenarios, capturing perceptual cues for precise prediction. Evaluated on the **Macao Connected and Autonomous Driving (MoCAD)** dataset, as well as **NGSIM** and **HighD** benchmarks, HLTP consistently outperforms existing models, especially in complex environments with incomplete data.  
For further details, visit the **[Project Page](https://github.com/Petrichor625/HLTP)**.

---

## 🧠 Framework

The HLTP’s teacher-student architecture involves:

- **Teacher Model**: Includes a Surround-Aware Encoder and Teacher Encoder, processing visual and context matrices to generate surround-aware and visual-aware vectors, which are then fed into the Teacher Multimodal Decoder. This enables the prediction of possible maneuvers for the target vehicle with associated probabilities.
- **Student Model**: Learned from the teacher model using a Knowledge Distillation Modulation (KDM) strategy, achieving accurate, human-like trajectory predictions even with minimal observational data.

 ![image](https://github.com/Petrichor625/HLTP/blob/main/HLTP/HLTP/pic/framework.png)

---

## ⚙️ Environment

- **Operating System**: Ubuntu 20.04
- **CUDA Version**: 11.3

---

## 🔧 Setup Instructions

1. **Creating the Conda Environment for HLTP**  
   Start by setting up a dedicated environment for HLTP:

   ```bash
   conda create --name HLTP python=3.7
   conda activate HLTP
   ```

2. **Installing PyTorch**  
   Install PyTorch with CUDA 11.3 compatibility:

   ```bash
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
   ```

3. **Installing Additional Requirements**  
   Finalize the environment setup with the required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️‍♂️ Training

1. **Train the Teacher Model**  
   To begin training the HLTP teacher model:

   ```bash
   python train_teacher.py
   ```

2. **Train the Student Model**  
   Train the student model using the pretrained teacher model:

   ```bash
   python train_student.py
   ```

---

## 📊 Evaluation

1. **Evaluate the Teacher Model**  
   Start evaluation for the teacher model:

   ```bash
   python evaluate_teacher.py
   ```

2. **Evaluate the Student Model**  
   Run the evaluation for the student model:

   ```bash
   python evaluate_student.py
   ```

---

## 📌 Citation

If you find our work useful in your research, please cite:

**A Cognitive-Based Trajectory Prediction Approach for Autonomous Driving**, published in the journal **_IEEE Transactions on Intelligent Vehicles_**.

```
@ARTICLE{10468619,
  author={Liao, Haicheng and Li, Yongkang and Li, Zhenning and Wang, Chengyue and Cui, Zhiyong and Li, Shengbo Eben and Xu, Chengzhong},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={A Cognitive-Based Trajectory Prediction Approach for Autonomous Driving}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Trajectory; Visualization; Brain modeling; Adaptation models; Predictive models; Decision making; Vehicle dynamics; Autonomous Driving; Trajectory Prediction; Cognitive Modeling; Knowledge Distillation; Interaction Understanding},
  doi={10.1109/TIV.2024.3376074}}
```

---

Thank you for exploring HLTP! If you have questions or need further assistance, feel free to reach out.

