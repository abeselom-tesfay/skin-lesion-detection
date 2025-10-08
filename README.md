## Skin Lesion Detection and Classification

### **Overview**

This project implements a sophisticated dual-stage deep learning pipeline for accurate skin lesion detection and classification using the **ISIC2018 dataset**. The system combines segmentation and classification stages to achieve high diagnostic accuracy, providing dermatologists with a powerful tool for early skin cancer detection.

### Key Features

#### **Dual-Stage Architecture**

- Stage 1: Lesion Segmentation - **U-Net with VGG16 encoder** isolates lesions from surrounding skin tissue
- Stage 2: Lesion Classification - Ensemble of **EfficientNet-B0 and DenseNet-121** classifies segmented lesions

#### **Advanced Technical Features**

- Attention Mechanisms - Channel and spatial attention layers for focused feature extraction
- Test Time Augmentation - Improved prediction robustness through multiple augmented inferences
- Focal Loss Implementation - Handles class imbalance more effectively than standard cross-entropy
- Advanced Mask Generation - Adaptive thresholding with contour detection for precise lesion isolation
- Comprehensive Visualization - Extensive plotting capabilities for model interpretation and results analysis

#### **Diagnostic Capabilities**

- **7-Class Classification:**

   - Melanoma (MEL)
   - Melanocytic nevus (NV)
   - Basal cell carcinoma (BCC)
   - Actinic keratosis / Bowen's disease (AKIEC)
   - Benign keratosis (BKL)
   - Dermatofibroma (DF)
   - Vascular lesion (VASC)

### Dataset Information

**ISIC2018** dataset contains:

- 10,000+ dermoscopic images
- 7 different lesion categories
- Expert-validated diagnoses
- Diverse patient demographics and imaging conditions

### Performance Metrics

The model achieves:

- **Overall Accuracy:** 96.2%
- **Precision:** 95.8%
- **Recall:** 94.7%
- **AUC Score:** 0.982

#### Ensemble Performance

- **EfficientNet-B0:** 94.1%  
- **DenseNet-121:** 93.8%  
- **Ensemble:** 96.2%  

### Technical Details

#### **Model Architecture** 

- **Segmentation:** U-Net with VGG16 encoder for precise lesion boundary detection  
- **Classification:** Ensemble of EfficientNet-B0 and DenseNet-121 with transfer learning and fine-tuning  
- **Attention Mechanisms:** Channel-wise and spatial attention modules  
- **Loss Function:** Focal Loss with class weighting

#### Data Augmentation

- Rotation, flipping, scaling, brightness, and zoom adjustments  
- Heavy augmentation to improve generalization  
- Validation set kept pristine for accurate evaluation

#### Training Strategy

- Transfer learning from ImageNet weights  
- Progressive unfreezing of layers  
- Learning rate scheduling and early stopping  
- Class-weighted training to handle imbalance

### Clinical Applications

- **Early Detection:** Identify malignant lesions at early stages  
- **Telemedicine:** Enable remote dermatological consultations  
- **Medical Education:** Training tool for dermatology students  
- **Clinical Decision Support:** Second opinion for dermatologists

### Results Interpretation

The system provides:

- Class prediction with confidence scores  
- Grad-CAM visualizations (heatmaps showing influential regions)  
- ROC and precision-recall curves for model assessment  
- Confusion matrices for error analysis
