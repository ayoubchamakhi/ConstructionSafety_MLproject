

# 🏗️ Real-Time Object Detection for Construction Workers using YOLOv9  
**A Machine Learning Project**  
*Supervised by Dr. Nabil Chaabene*  
🔗 [Project Presentation](https://www.canva.com/design/DAGCGxwJQpo/AE-u1nBwy_46SF7qF8ZLJQ/view?utm_content=DAGCGxwJQpo&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hdfef3dbaf6))

### 📌 Project Overview  
This project focuses on implementing **real-time object detection** to ensure the safety of construction workers by identifying personal protective equipment (PPE) in live video streams. The solution leverages **YOLOv9 (You Only Look Once)** – a state-of-the-art object detection algorithm that detects objects within images and video feeds efficiently.

---

## 📋 Table of Contents  
- [Project Overview](#-project-overview)  
- [Key Features](#-key-features)  
- [How It Works](#-how-it-works)  
- [Technologies & Tools](#-technologies--tools)  
- [Model Performance](#-model-performance)  
- [Installation & Setup](#-installation--setup)  
- [Live Demonstration](#-live-demonstration)  
- [Future Improvements](#-future-improvements)  
- [References](#-references)  

---

## 🚀 Key Features  
- **Real-time Detection**: Capable of detecting workers' PPE on live camera streams and video files.  
- **High Accuracy**: Utilizes **YOLOv9**, ensuring faster and more precise detections.  
- **Hyperparameter Tuning**: Fine-tuned epochs and learning rates for optimal model performance.  
- **Live Testing**: Deployable on local environments and cloud-based systems (e.g., Google Colab).  
- **Scalable**: Compatible with CPUs, GPUs, and edge devices.  

---

## ⚙️ How It Works  
1. **Dataset Acquisition**: Labeled dataset of construction workers wearing safety gear.  
2. **Model Training**: Fine-tuned **YOLOv9** on the dataset using **Google Colab**.  
3. **Hyperparameter Tuning**: Adjusted training epochs to balance model performance and avoid overfitting/underfitting.  
4. **Real-Time Detection**:  
   - Implemented real-time object detection using a webcam.  
   - Deployed the trained model in **PyCharm** and executed detection scripts locally.  
5. **Evaluation**: Performance assessed through **mAP (Mean Average Precision)**, **Precision**, and **Recall** metrics.  

---

## 🛠️ Technologies & Tools  
- **Machine Learning Frameworks**: TensorFlow, PyTorch  
- **Model**: YOLOv9 (You Only Look Once)  
- **Languages**: Python  
- **Development Tools**:  
   - **Google Colab** – Model training environment  
   - **PyCharm** – Local development and testing  
- **Libraries**: OpenCV, NumPy, Matplotlib  
- **Dataset**: Custom-labeled images of construction workers  
- **Version Control**: Git & GitHub  

---

## 📊 Model Performance  
- **Metrics**:  
   - **mAP50**: Evaluates precision at 50% Intersection over Union (IoU) threshold.  
   - **mAP50-95**: Measures performance across IoU thresholds from 0.50 to 0.95.  
   - **Precision**: Measures the accuracy of object detection (True Positives).  
   - **Recall**: Reflects the ability to detect all relevant objects.  

- **Graph Interpretation**:  
   - **Training Loss vs Validation Loss** helps detect overfitting.  
   - A gradual decrease in loss indicates model improvement.  

---

## 📥 Installation & Setup  
### 1. Clone the Repository  
```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```  

### 2. Install Required Libraries  
```bash
pip install -r requirements.txt
```  

### 3. Model Training (Google Colab)  
- Upload the dataset to Google Colab.  
- Use the provided `.ipynb` notebook to train the model.  
- Adjust the number of epochs based on validation performance.  

### 4. Real-Time Testing (PyCharm)  
```bash
python detect.py --source 0  # 0 for webcam, or path to video file
```  

---

## 🎥 Live Demonstration  
- Real-time detection executed via webcam or video input.  
- Model deployed on local machines and capable of running on GPUs for faster inference.  

---

## 🔧 Future Improvements  
- **Fine-Tuning Small Object Detection**: Improve model accuracy for detecting smaller PPE items.  
- **Model Lightweighting**: Optimize model for edge devices.  
- **Dataset Expansion**: Increase the diversity of datasets for better generalization.  
- **Multi-Class Detection**: Expand to detect different types of PPE simultaneously.  

---

## 📚 References  
- [Ultralytics GitHub - YOLOv5s6 Training Results](https://github.com/ultralytics/yolov5/issues/8185)  
- [CVPR Paper - You Only Look Once: Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)  
- [YouTube - Neural Network Deep Learning](https://www.youtube.com/watch?v=aircAruvnKk)  
- [YouTube - Object Detection 101 Course](https://www.youtube.com/watch?v=WgPbbWmnXJ8)  
- [YouTube - How YOLO Works](https://www.youtube.com/watch?v=svn9-xV7wjk)  
