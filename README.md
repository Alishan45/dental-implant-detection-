# 🦷 Dental Implant Detection with YOLOv11m  

**A high-precision object detection model for identifying dental implants in radiographic images, fine-tuned on a custom dataset using YOLOv11m.**  

<div align="center">
  <img src="https://media.giphy.com/media/QB17GCGiD97t5STPNg/giphy.gif?cid=ecf05e47kkgngq7ebfr55xj3jp0xxucaqbn7k28k4mhayu6g&ep=v1_gifs_search&rid=giphy.gif&ct=g" alt="Dental-implant"/>
</div>

---

## 📌 Overview  
This project leverages **YOLOv11m**, a state-of-the-art object detection architecture, to detect and classify dental implants in panoramic or periapical radiographs. The model is fine-tuned on a proprietary dataset to achieve high accuracy in identifying implant types, positions, and potential anomalies, supporting dental diagnostics and treatment planning.  

**Key Applications**:  
- Automated implant inventory tracking  
- Pre-surgical planning assistance  
- Anomaly detection (e.g., misalignment, fractures)  

---

## ✨ Features  
- **YOLOv11m Fine-Tuning**: Optimized for small but critical features in dental radiographs.  
- **Custom Dataset**: Annotated dental implants (bounding boxes + classes) from clinical sources.  
- **Inference Pipeline**: Ready-to-use scripts for integration with DICOM viewers or PACS systems.  
- **Performance Metrics**: mAP@0.5, precision-recall curves, and confusion matrices.  
- **Modular Code**: Easily adaptable to other medical object detection tasks.  

---

## ⚙️ Installation  

### Prerequisites  
- Python 3.8+  
- PyTorch 2.0+  
- CUDA 11.7 (for GPU acceleration)  

### Steps  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/dental-implant-detection.git  
   cd dental-implant-detection  
   ```  

2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Download weights (or train your own):  
   ```bash  
   wget https://example.com/path/to/yolov11m_implant_weights.pt  
   ```  

---

## 🚀 Usage  

### Training  
1. Place your dataset in `data/` with annotations in YOLO format.  
2. Update `configs/implant_config.yaml` with dataset paths and hyperparameters.  
3. Run training:  
   ```bash  
   python train.py --cfg configs/implant_config.yaml --weights yolov11m.pt  
   ```  

### Inference  
Run detection on a single image:  
```bash  
python detect.py --source data/test/image_001.png --weights yolov11m_implant_weights.pt  
```  

### Export to ONNX/TensorRT  
```bash  
python export.py --weights yolov11m_implant_weights.pt --include onnx  
```  

---

## 📊 Results  
| Model          | mAP@0.5 | Precision | Recall |  
|----------------|---------|-----------|--------|  
| YOLOv11m (Ours)| 0.92    | 0.89      | 0.91   |  
| YOLOv8         | 0.85    | 0.82      | 0.83   |  

![Confusion Matrix](assets/confusion_matrix.png)  

---

## 🤝 Contributing  
Contributions are welcome! Please:  
1. Fork the repository.  
2. Create a branch (`git checkout -b feature/your-feature`).  
3. Submit a PR with detailed explanations.  

---

## 📜 License  
This project is licensed under **MIT**. For commercial use, please contact the author.  

---

## 📬 Contact  
For questions or collaborations:  
- **Email**: Ali3819381@gmail.com  
---  
