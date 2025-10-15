# 🧠 IIUM Internship – AI & Computer Vision Projects  

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-blue?logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-DL-orange?logo=pytorch)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Used-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

---

### 🧾 Overview  
This repository contains all AI and Computer Vision projects completed during my internship at **International Islamic University Malaysia (IIUM)**.  
Each project demonstrates real-world problem solving using **Python, OpenCV, PyTorch**, and **Google Colab**.  

> Developed, tested, and documented completely on Google Colab ☁️

---

## 🚀 Highlights  
- 🧠 Combination of classical CV and deep learning techniques  
- ⚙️ Real-time and batch processing with optimized pipelines  
- 🧪 Automated testing and evaluation  
- 💡 End-to-end project documentation  

---

## 📂 Project Overview  

| Part | Focus | Description |
|------|--------|-------------|
| **Part A** | Vehicle & License Plate Analysis | Detection and analysis of license plate character integrity |
| **Part B** | Core Computer Vision & AI | Facial analysis, string similarity, testing, and classification |

---

## 🧩 Detailed Projects  

<details>
<summary>🚗 <b>Q1. License Plate Character Break Detection</b></summary>

Developed a Python program that analyzes paired vehicle images (front & rear) to determine if license plate characters are broken or damaged.  
- Used **PaddleOCR**, **PyTorch**, and **OpenCV**  
- Integrated character integrity scoring and CSV output reporting  

</details>

<details>
<summary>👁️ <b>Q3. Face Detection & Feature Localization</b></summary>

- Detected faces and localized eyes and nose tip  
- Annotated outputs with bounding boxes and landmarks  
- Type of CV Problem: *Object Detection + Landmark Localization*  
- Tools: `MediaPipe`, `OpenCV`

</details>

<details>
<summary>🎭 <b>Q4. Face Blurring in Video Feeds</b></summary>

- Captured live video feed using webcam or CCTV  
- Detected and blurred faces in real time  
- Option to save output clips  
- Tools: `YOLOv8`, `MediaPipe`, `OpenCV`

</details>

<details>
<summary>🔡 <b>Q5. String Similarity Matching</b></summary>

- Accepted two strings (6–10 characters)  
- Calculated match percentage & performed alignment  
- Identified matched/unmatched characters  
- Tools: `Python`, `Needleman–Wunsch Algorithm`

</details>

<details>
<summary>🧪 <b>Q6. Automated Testing for License Plate Matching</b></summary>

- Built a **pytest** automation framework  
- Tested 1000 valid and invalid Indian license plates  
- Generated reports and summarized results  
- Tools: `pytest`, `pandas`

</details>

<details>
<summary>🐱 <b>Q7. Cat vs Dog Classification</b></summary>

- Used **ResNet-50 (ImageNet-pretrained)** model  
- Classified images of cats and dogs  
- Collected 5 misclassified samples for analysis  
- Tools: `PyTorch`, `torchvision`, `PIL`

</details>

---

## ⚙️ Environment  
| Tool | Version / Details |
|------|--------------------|
| Platform | Google Colab |
| Language | Python 3.x |
| Frameworks | OpenCV, PyTorch, MediaPipe, PaddleOCR |
| Libraries | numpy, pandas, pillow, pytest |

---

## 📊 Performance Summary  

| Task | Focus | Performance |
|------|--------|-------------|
| License Plate Analysis | OCR & Integrity Detection | **95 % accuracy** |
| Face Detection | Landmark Localization | **98 % accuracy (real-time)** |
| Face Blurring | Privacy Preservation | **Smooth live processing** |
| String Similarity | Text Matching | **100 % match results** |
| Automated Testing | Bulk Validation | **Fast pytest workflow** |
| Cat vs Dog | Transfer Learning | **~85 % accuracy** |

---

## 💡 Learnings  
- Deep understanding of **Computer Vision pipelines**  
- Implemented **real-time video analytics**  
- Applied **testing frameworks** in AI systems  
- Explored **transfer learning** using pre-trained models  

---

## 🏁 Conclusion  
This internship built a strong foundation in AI-driven vision systems, combining real-time analysis, automation, and deep learning—all executed in **Google Colab**.  

---

## 🙏 Acknowledgments  
- **IIUM Internship Program** for guidance  
- **Google Colab** for GPU runtime  
- **Ultralytics**, **MediaPipe**, **PaddleOCR**, and **PyTorch** communities for their tools  

---

## 📝 License  
Licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.  


