# Telco Customer Churn Prediction Using Artificial Neural Network

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

A comprehensive Deep Learning project that predicts customer churn using an Artificial Neural Network (ANN). The model achieves **84% AUC score** with **79% recall** for identifying at-risk customers.

## 🎯 Project Overview

This project analyzes telecommunications customer data to predict which customers are likely to churn (cancel their service). The solution uses Deep Learning techniques to help businesses identify at-risk customers and take proactive retention measures.

### Key Features
- **Deep Learning Model**: 5-layer ANN with Batch Normalization and Dropout
- **Robust Preprocessing**: Handles mixed data types and class imbalance
- **High Performance**: 84.2% AUC score with optimized Recall (79%)
- **Production Ready**: Serialized models and preprocessors for deployment

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **AUC** | 0.842 |
| **Accuracy** | 75% |
| **Precision (Churn)** | 52% |
| **Recall (Churn)** | **79%** |
| **F1-Score (Churn)** | 0.63 |

## 🗂️ Project Structure

```
customer-churn-ann-classifier/
│
├── customer_churn.ipynb          # Main Jupyter notebook with complete analysis
├── customer_churn.pdf            # PDF export of the notebook
├── churn_ann_model.keras         # Trained ANN model
├── churn_preprocessor.pkl        # Data preprocessing pipeline
├── churn_threshold.pkl           # Optimized classification threshold
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

### Usage

1. **Clone the repository:**
```bash
git clone https://github.com/adeel-iqbal/customer-churn-ann-classifier.git
cd customer-churn-ann-classifier
```

2. **Load and use the trained model:**
```python
import joblib
import tensorflow as tf
import pandas as pd

# Load saved components
model = tf.keras.models.load_model('churn_ann_model.keras')
preprocessor = joblib.load('churn_preprocessor.pkl')
threshold = joblib.load('churn_threshold.pkl')

# Prepare your data
new_data = pd.DataFrame({...})  # Your customer data
X_processed = preprocessor.transform(new_data)

# Make predictions
predictions = model.predict(X_processed)
churn_predictions = (predictions >= threshold).astype(int)
print(churn_predictions)
```

3. **Run the complete analysis:**
```bash
jupyter notebook customer_churn.ipynb
```

## 📈 Dataset Information

The project uses the **Telco Customer Churn** dataset with the following characteristics:

- **Samples**: 7,043 customers
- **Features**: 20 attributes (after removing customerID)
- **Target**: Binary classification (Churn: Yes/No)
- **Class Distribution**: 
  - No Churn: 5,164 (73.4%)
  - Churn: 1,857 (26.4%)

### Key Features Include:
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Account Info**: Tenure, Contract type, Payment method
- **Services**: Phone, Internet, Online security, Tech support
- **Financial**: Monthly charges, Total charges

## 🔧 Model Architecture

The network is a 5-layer feedforward ANN with ~121 total neurons, using ReLU activations, Batch Normalization, Dropout, and a Sigmoid output.

```
Input Layer (45 features after preprocessing)
    ↓
Dense Layer (64 neurons, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (32 neurons, ReLU) + BatchNorm
    ↓
Dense Layer (16 neurons, ReLU) + BatchNorm
    ↓
Dense Layer (8 neurons, ReLU) + BatchNorm
    ↓
Output Layer (1 neuron, Sigmoid)
```

### Key Training Features:
- **Optimizer**: Adam (Learning Rate = 0.005)
- **Loss Function**: Binary Crossentropy
- **Class Balancing**: Computed class weights for imbalanced data
- **Regularization**: Dropout (0.3) and Batch Normalization
- **Early Stopping**: Prevents Overfitting with Patience=5

## 📋 Methodology

1. **Data Preprocessing**
   - Handle missing values and data type conversions
   - Remove duplicates (22 found)
   - Feature Scaling (StandardScaler) for Numerical Features
   - One-Hot Encoding for Categorical Features

2. **Exploratory Data Analysis**
   - Distribution analysis of target variable
   - Feature correlation with churn
   - Visualization of key patterns

3. **Model Development**
   - Train-test split (80/20) with stratification
   - Neural Network architecture design
   - Hyperparameter optimization
   - Class weight balancing for imbalanced data

4. **Model Evaluation**
   - Confusion matrix analysis
   - ROC-AUC curve assessment
   - Precision-recall trade-off analysis

## 🎯 Business Impact

This model helps businesses:
- **Identify At-Risk Customers**: 79% recall ensures most churners are caught
- **Optimize Retention Efforts**: Focus resources on high-probability churn cases
- **Reduce Revenue Loss**: Proactive intervention before customer cancellation
- **Improve Customer Experience**: Address pain points leading to churn

## 📞 Contact

**Adeel Iqbal Memon**
- 📧 Email: adeelmemon096@yahoo.com
- 💼 LinkedIn: [linkedin.com/in/adeeliqbalmemon](https://linkedin.com/in/adeeliqbalmemon)
- 🐱 GitHub: [github.com/adeel-iqbal](https://github.com/adeel-iqbal)

## 🙏 Acknowledgments

- Dataset: IBM Sample Data Sets
- TensorFlow/Keras for Deep Learning framework
- Scikit-learn for Preprocessing and Evaluation Metrics
- Seaborn/Matplotlib for Data Visualization

---

⭐ **If you found this project helpful, please give it a star!** ⭐
