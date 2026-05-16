# Brain Tumor Detection and Classification

A deep learning project for detecting and classifying brain tumors from MRI images using Convolutional Neural Networks (CNN) and Transfer Learning with VGG16. This project uses Python, TensorFlow/Keras, and advanced image processing techniques to achieve accurate tumor classification.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [What You'll Find Here](#whats-included)
- [Dataset Information](#dataset-information)
- [Installation & Setup](#installation--setup)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Key Implementation Details](#key-implementation-details)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Results & Performance](#results--performance)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project implements a complete end-to-end machine learning pipeline for:

- **Loading MRI Brain Scan Images** from Google Drive
- **Data Visualization & Exploration** - Displaying sample MRI images
- **Data Preprocessing** - Image resizing and normalization
- **Model Building** - Using VGG16 Transfer Learning for feature extraction
- **Model Training** - Training a custom neural network on extracted features
- **Evaluation & Testing** - Assessing model performance on test data
- **Predictions** - Making predictions on new brain MRI images

The project achieves high accuracy in detecting and classifying different types of brain tumors from MRI scans, making it a valuable tool for medical imaging analysis.

---

## 📦 What's Included

### Main Files:
- **SDP_Project.ipynb** - Complete Jupyter Notebook with the entire project pipeline
- **README.md** - This documentation file

---

## 📊 Dataset Information

### Data Structure:
```
MRI_Images/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── pituitary_tumor/
│   └── no_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── pituitary_tumor/
    └── no_tumor/
```

### Tumor Classes:
1. **Glioma Tumor** - Primary brain tumors originating from glial cells
2. **Meningioma Tumor** - Tumors originating from the meninges (brain membranes)
3. **Pituitary Tumor** - Tumors in the pituitary gland
4. **No Tumor** - Normal/healthy brain scans

### Data Characteristics:
- High-resolution MRI brain scan images
- Multiple tumor classifications
- Separate training and testing datasets
- Stratified class distribution

---

## 🚀 Installation & Setup

### Prerequisites:
```
- Python 3.7+
- Google Colab (or local Jupyter Notebook)
- Google Drive (for dataset storage)
- GPU support (recommended for faster training)
```

### Required Libraries:
```python
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn pillow
```

### Step-by-Step Setup:

1. **Clone the Repository:**
```bash
git clone https://github.com/Ritupriya176/Brain-Tumor-Detection-and-Classification.git
cd Brain-Tumor-Detection-and-Classification
```

2. **Setup Google Drive Access:**
   - Upload your MRI dataset to Google Drive under: `MyDrive/MRI_Images/`
   - Organize as: `Training/` and `Testing/` folders with tumor class subfolders

3. **Open in Google Colab:**
   - Upload `SDP_Project.ipynb` to Google Colab
   - Connect to Google Drive when prompted

4. **Run the Notebook:**
   - Execute cells sequentially to run the complete pipeline

---

## 🔄 Project Workflow

### Phase 1: Environment Setup
- Import all required Python libraries
- Mount Google Drive to access the dataset

### Phase 2: Data Loading
- Load training images from organized directories
- Load testing images from organized directories
- Shuffle datasets for randomization
- Store image paths and labels

### Phase 3: Data Exploration
- Visualize random MRI brain scans
- Display 10 sample images in a 2x5 grid
- Show image labels for each sample
- Understand data distribution

### Phase 4: Data Preprocessing
- Resize images to 224x224 pixels (VGG16 input size)
- Normalize pixel values (0-255 to 0-1)
- Apply data augmentation techniques
- Prepare data for model training

### Phase 5: Feature Extraction
- Load pre-trained VGG16 model (trained on ImageNet)
- Extract features using VGG16 without classification layer
- Use transfer learning to leverage pre-learned features
- Reduce dimensionality of images

### Phase 6: Model Development
- Build custom neural network on top of VGG16 features
- Architecture:
  - Input layer for VGG16 features
  - Dense layer with 128 neurons
  - Dropout layer (0.5) for regularization
  - Dense layer with 64 neurons
  - Dropout layer (0.5) for regularization
  - Output layer with softmax for 4-class classification

### Phase 7: Model Training
- Optimizer: Adam (adaptive learning rate)
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Track training history (loss and accuracy over epochs)

### Phase 8: Model Evaluation
- Evaluate on test dataset
- Calculate accuracy, precision, recall, F1-score
- Generate confusion matrix
- Analyze per-class performance

### Phase 9: Predictions & Visualization
- Make predictions on new images
- Display confidence scores
- Visualize prediction results

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3** | Programming Language |
| **TensorFlow/Keras** | Deep Learning Framework |
| **VGG16** | Pre-trained CNN for feature extraction |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |
| **Scikit-learn** | ML utilities (shuffling, metrics) |
| **Pillow (PIL)** | Image processing |
| **Matplotlib** | Data visualization |
| **Google Colab** | Cloud-based execution environment |
| **Google Drive** | Dataset storage |

---

## 🧠 Model Architecture

### Transfer Learning Approach:

```
Input Image (224x224x3)
         ↓
    VGG16 (Pre-trained)
    [Extract Features]
         ↓
Feature Vector (512-dim)
         ↓
Dense Layer (128 neurons) + ReLU
         ↓
Dropout (0.5)
         ↓
Dense Layer (64 neurons) + ReLU
         ↓
Dropout (0.5)
         ↓
Output Layer (4 neurons) + Softmax
         ↓
Classification [0-3]: glioma, meningioma, pituitary, no_tumor
```

### Why VGG16?
- **Pre-trained weights** on ImageNet dataset
- **Excellent feature extraction** for medical images
- **Transfer learning** reduces training time significantly
- **Proven architecture** for image classification tasks

### Regularization Techniques:
- **Dropout (0.5)**: Prevents overfitting by randomly deactivating neurons
- **Adam Optimizer**: Adaptive learning rates for faster convergence
- **Train/Test Split**: Separate data for unbiased evaluation

---

## 🔍 Key Implementation Details

### 1. Data Loading:
```python
# Iterate through directories and load image paths
train_dir = '/content/drive/MyDrive/MRI_Images/Training'
test_dir = '/content/drive/MyDrive/MRI_Images/Testing'

# Create lists of paths and corresponding labels
train_paths, train_labels = [...], [...]
test_paths, test_labels = [...], [...]

# Shuffle for randomization
train_paths, train_labels = shuffle(train_paths, train_labels)
```

### 2. Image Visualization:
- Display 10 random training images
- Show in 2x5 grid format
- Display corresponding labels
- Image size: 224x224 pixels

### 3. VGG16 Integration:
```python
from tensorflow.keras.applications import VGG16

# Load pre-trained VGG16
base_model = VGG16(weights='imagenet', include_top=False)

# Extract features from images
features = base_model.predict(images)
```

### 4. Custom Model:
```python
model = Sequential([
    Input(shape=(512,)),  # VGG16 feature dimension
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 tumor classes
])
```

---

## 📈 Data Processing Pipeline

```
Raw MRI Images
    ↓
Load from Google Drive
    ↓
Organize into train/test sets
    ↓
Shuffle for randomization
    ↓
Resize to 224x224 pixels
    ↓
Normalize pixel values (0-1)
    ↓
VGG16 Feature Extraction
    ↓
Custom Neural Network Training
    ↓
Model Evaluation & Testing
    ↓
Generate Predictions & Metrics
```

---

## 📊 Results & Performance

### Model Metrics:
- **Accuracy**: Percentage of correct predictions on test set
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows classification performance per tumor class

### Performance Analysis:
- Per-class accuracy for each tumor type
- Misclassification patterns
- Model strengths and weaknesses
- Overall classification reliability

*See notebook output for specific metrics and visualizations*

---

## 💻 How to Run

### Step 1: Open in Google Colab
```
1. Go to Google Colab: https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Select SDP_Project.ipynb
4. OR Open directly: 
   https://colab.research.google.com/github/Ritupriya176/Brain-Tumor-Detection-and-Classification/blob/main/SDP_Project.ipynb
```

### Step 2: Setup Dataset
```
1. Create MRI_Images folder in Google Drive
2. Upload Training and Testing folders
3. Organize by tumor class:
   - glioma_tumor/
   - meningioma_tumor/
   - pituitary_tumor/
   - no_tumor/
```

### Step 3: Run Cells
```
1. Mount Google Drive (when prompted)
2. Execute cells in sequence
3. Monitor training progress
4. Review results and visualizations
```

### Step 4: Make Predictions
```python
# Load a new image
new_image = load_img('path/to/image.jpg', target_size=(224, 224))

# Preprocess
image_array = np.array(new_image) / 255.0

# Extract features using VGG16
features = base_model.predict(image_array)

# Make prediction
prediction = model.predict(features)

# Get class label
tumor_class = labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100
```

---

## 📁 Project Structure

```
Brain-Tumor-Detection-and-Classification/
│
├── README.md                      # Project documentation (this file)
├── SDP_Project.ipynb              # Main Jupyter notebook with complete pipeline
│
├── Data Structure (in Google Drive):
│   └── MRI_Images/
│       ├── Training/
│       │   ├── glioma_tumor/      # ~1000+ images
│       │   ├── meningioma_tumor/  # ~500+ images
│       │   ├── pituitary_tumor/   # ~500+ images
│       │   └── no_tumor/          # ~500+ images
│       │
│       └── Testing/
│           ├── glioma_tumor/      # ~100+ images
│           ├── meningioma_tumor/  # ~100+ images
│           ├── pituitary_tumor/   # ~100+ images
│           └── no_tumor/          # ~100+ images
│
└── Output:
    ├── Trained Model (saved in Colab)
    ├── Training History (plots)
    ├── Predictions (on test set)
    └── Performance Metrics (accuracy, precision, recall, F1)
```

---

## 🎓 Learning Outcomes

By studying this project, you'll learn:

✅ **Transfer Learning** - Leveraging pre-trained models (VGG16)
✅ **Medical Image Processing** - Working with MRI scans
✅ **Deep Learning** - Building and training neural networks
✅ **Data Pipeline** - Organizing and processing large image datasets
✅ **Model Evaluation** - Using appropriate metrics for classification
✅ **Google Colab & Drive** - Cloud-based ML development
✅ **Python & TensorFlow** - Practical deep learning implementation

---

## 🤝 Contributing

We welcome contributions! To improve this project:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Make** your changes and improvements
4. **Commit** with clear messages (`git commit -am 'Add improvement'`)
5. **Push** to your branch (`git push origin feature/improvement`)
6. **Open** a Pull Request

### Suggested Improvements:
- Experiment with other pre-trained models (ResNet, DenseNet, EfficientNet)
- Implement data augmentation techniques
- Add real-time prediction interface
- Improve model accuracy with hyperparameter tuning
- Create a web application for predictions
- Add grad-CAM visualization for model interpretability

---

## 📄 License

This project is open source and available under the **MIT License**. Feel free to use, modify, and distribute this code for educational and research purposes.

---

## 📧 Contact & Support

For questions, issues, or suggestions:
- Open an **Issue** on GitHub
- Contact the project maintainer: [Ritupriya176](https://github.com/Ritupriya176)
- Check existing discussions for solutions

---

## ⚖️ Disclaimer

**Important:** This project is for **educational and research purposes only**. 

- It should NOT be used for actual medical diagnosis without clinical validation
- Always consult with qualified medical professionals for clinical decisions
- Medical imaging analysis requires validation by radiologists and medical experts
- This tool is meant to demonstrate ML techniques, not replace professional medical advice

---

## 🔗 Additional Resources

- **TensorFlow Documentation**: https://www.tensorflow.org/api_docs
- **Keras Guide**: https://keras.io/
- **VGG16 Paper**: https://arxiv.org/abs/1409.1556
- **Medical Image Analysis**: https://en.wikipedia.org/wiki/Medical_image_computing_and_computer-assisted_intervention
- **Transfer Learning Guide**: https://cs231n.github.io/transfer-learning/

---

## 📝 Citation

If you use this project in your research or work, please cite:

```
@github{Brain-Tumor-Detection-Classification,
  author = {Ritupriya176},
  title = {Brain Tumor Detection and Classification},
  year = {2026},
  url = {https://github.com/Ritupriya176/Brain-Tumor-Detection-and-Classification}
}
```

---

**Last Updated:** May 16, 2026

**Status:** Active & Maintained ✅

Happy Learning! 🚀