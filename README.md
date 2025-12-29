# 🥊 AI-Powered Boxing Analytics System

An **AI-driven Boxing Performance Analytics System** that analyzes recorded boxing videos using **Computer Vision**, **Pose Estimation**, and **custom motion analysis algorithms**.  
The system automatically detects punches, evaluates technique, and visualizes performance metrics through a cinematic, real-time HUD overlay.

🔗 **Repository:** https://github.com/supunabeywickrama/Boxing-Analytics

---
## 🎥 Demo Video

[![Watch Demo](https://img.shields.io/badge/▶%20Watch%20Demo-Video-red?style=for-the-badge)](https://drive.google.com/file/d/1s12hjDeaD2il69cgqw3kLnKj4SG5rIWw/view?usp=sharing)

---

## 🚀 Project Overview

The **AI-Powered Boxing Analytics System** is a computer vision–based performance analysis platform designed to extract meaningful insights from recorded boxing training videos.  
The system analyzes an athlete’s movements frame-by-frame to detect punches, evaluate technique, measure performance intensity, and visualize results through a cinematic real-time overlay.

Unlike traditional motion-capture or wearable-based systems, this project works using **only standard video input**, making advanced boxing analytics more accessible and cost-effective.

At its core, the system combines **MediaPipe Pose Estimation** for precise body landmark tracking with **YOLOv8 Object Detection** for robust person localization. Using these signals, a custom motion-analysis pipeline classifies punch types (Jab, Cross, Hook, Uppercut) and computes key performance metrics such as speed, power index, technique quality, energy (punches per minute), and fatigue level.

The output is not only numerical data but also a **visually rich annotated video**, featuring pose skeletons, punch labels, animated performance bars, motion trails, particle effects, and energy meters. In parallel, the system automatically generates structured analytics files (CSV timelines and summaries) that can be used for further analysis, reporting, or integration into other applications.

This project demonstrates how modern AI and computer vision techniques can be applied to **sports analytics, training optimization, and performance monitoring** without specialized hardware.

---

## ✨ Key Features

- 🥊 **Punch Detection & Classification**
  - Jab
  - Cross
  - Hook
  - Uppercut

- 📊 **Performance Metrics**
  - Speed Index (normalized wrist velocity)
  - Power Index (acceleration-based)
  - Technique Score (elbow–shoulder coordination)
  - Energy Level (Punches Per Minute – PPM)
  - Fatigue Simulation (decay-based model)

- 🔁 **Combo Recognition**
  - Detects multi-punch sequences within time windows

- 🎥 **Cinematic Visualization**
  - Pose skeleton overlay
  - Motion trails on wrists
  - Particle bursts on impact
  - Animated performance bars
  - Energy VU meter
  - Punch label banners

- 📁 **Automated Outputs**
  - Annotated video (`_annotated.mp4`)
  - Per-punch CSV logs
  - Per-second timeline analytics
  - Session-level summary reports

---

## 🧠 System Pipeline

1. **Video Input**
   - Reads saved boxing videos frame-by-frame

2. **Pose Estimation**
   - MediaPipe detects 33 body landmarks

3. **Person Localization (Optional)**
   - YOLOv8 crops the boxer ROI for improved accuracy

4. **Punch Classification**
   - Uses wrist velocity, elbow angles, and motion direction

5. **Metrics Computation**
   - Speed, Power, Technique, Energy, Fatigue

6. **Visualization & Export**
   - Real-time HUD overlay + CSV analytics

---

## 🛠️ Tech Stack

- **Programming:** Python  
- **Computer Vision:** OpenCV, MediaPipe Pose  
- **Deep Learning:** YOLOv8 (Ultralytics), TensorFlow Lite  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** OpenCV HUD, Particle FX engine  
- **Version Control:** Git, GitHub

---

## 📂 Project Structure

<img width="524" height="572" alt="image" src="https://github.com/user-attachments/assets/d764c7ea-f799-43b5-9119-37b19f069f80" />

---

## 📂 Project Structure

