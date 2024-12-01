# DPaA Project: Group-2
## Badges

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## NASA Asteroid Data Analysis 

## **Introduction**

This project aims to analyze NASA's asteroid dataset (`nasa.csv`) to identify potential patterns and provide insights through clustering and classification models. It is designed to preprocess, analyze, and model the dataset effectively, offering a streamlined workflow for similar datasets in astronomy or related fields.
---

## **Overview**

### **What does the program do?**

This program processes and analyzes a dataset of near-Earth objects (NEOs) to:
1. Clean and preprocess data by handling missing values, encoding target labels, and removing unnecessary features.
2. Visualize data distributions and correlations for exploratory data analysis (EDA).
3. Reduce dimensionality using techniques like PCA, t-SNE, and UMAP.
4. Perform clustering and classification tasks with models such as Random Forest, Logistic Regression, SVM, K-Means, DBSCAN, and others.
5. Evaluate model performance with metrics like accuracy, precision, recall, and F1 score.

### **Features**

- Data cleaning and preprocessing for high-dimensional datasets.
- Visualization of correlations and feature distributions.
- Feature selection using Random Forest and variance thresholding.
- Dimensionality reduction techniques for better visualization.
- Implementation of clustering and classification models.
- Evaluation and comparison of model performance metrics.

---

## **Dataset Description**

The dataset, `nasa.csv`, includes features such as orbital details, sizes, and labels indicating whether an asteroid is potentially hazardous. Key details:
- **Target Variable**: `Hazardous` (binary classification: 0 for True, 1 for False).
- **Features**: Includes size, orbital details, and velocity-related data.
- **Removed Features**: IDs, dates, and other metadata irrelevant to the analysis.

---

## **Training and Testing**

### **Preprocessing**

1. **Data Cleaning**:
   - Removed irrelevant columns (e.g., `Name`, `Neo Reference ID`).
   - Renamed `Hazardous` to `Class` and encoded it as binary (0/1).
   - Eliminated highly correlated features using a correlation matrix.

2. **Dimensionality Reduction**:
   - Applied PCA, t-SNE, and UMAP for feature reduction and visualization.

3. **Feature Selection**:
   - Used variance thresholding to remove low-variance features.
   - Selected the top 10 most important features using Random Forest.

### **Model Training**

The following models were trained:
- **Random Forest**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

Hyperparameters were optimized using `GridSearchCV` and cross-validation.

#### **Additional Techniques**
1. **L1 and L2 Regularization**:
   - Applied to Logistic Regression to improve performance.
   - Best parameters: `{C: 100, penalty: 'l1', solver: 'liblinear'}`.

2. **Dimensionality Reduction**:
   - Models were tested with features transformed using PCA, t-SNE, and UMAP.

### **Testing and Validation**

Models were evaluated using Stratified K-Fold cross-validation. Metrics include:
- Accuracy
- Precision
- Recall
- F1 Score

---

## **Best Results**

### **Overall Model Performance (All Features)**

| Model               | Accuracy  | Precision | Recall    | F1 Score  |
|---------------------|-----------|-----------|-----------|-----------|
| **Random Forest**   | **0.9964** | **0.9980** | **0.9977** | **0.9978** |

Random Forest achieved the highest overall performance using all features, with near-perfect precision, recall, and F1 score.

---

### **Regularization Results (Logistic Regression)**

| Metric       | Best Parameters         | Accuracy | Precision | Recall    | F1 Score  |
|--------------|-------------------------|----------|-----------|-----------|-----------|
| Logistic Regression | `{C: 100, penalty: 'l1', solver: 'liblinear'}` | 0.9552   | 0.9733    | 0.9733    | 0.9733    |

L1 regularization marginally improved Logistic Regression's performance.

---

### **Dimensionality Reduction Results**

| Reduction Method | Model               | Accuracy  | Precision | Recall    | F1 Score  |
|------------------|---------------------|-----------|-----------|-----------|-----------|
| **All Features** | **Random Forest**   | **0.9964** | **0.9980** | **0.9977** | **0.9978** |
| **PCA**          | Logistic Regression | 0.8336    | 0.8435    | **0.9842** | 0.9084    |
| **t-SNE**        | Random Forest       | 0.9221    | 0.9474    | 0.9606    | 0.9539    |
| **UMAP**         | Random Forest       | 0.8748    | 0.9145    | 0.9385    | 0.9263    |

Dimensionality reduction techniques like PCA, t-SNE, and UMAP showed moderate success, but Random Forest consistently delivered the best performance.

---

## **Conclusion**

1. **Best Overall Model**: Random Forest using all features.
2. **Best Dimensionality Reduction**: Random Forest with t-SNE features for balanced performance.
3. **Regularization**: L1 regularization slightly improved Logistic Regression's metrics.

Random Forest stands out as the most effective model across all metrics, making it the top choice for this dataset.

---

## **Model Parameters**

- **Random Forest**:
  - `n_estimators`: 100
  - `max_depth`: 20
  - `random_state`: 42
- **Logistic Regression**:
  - `C`: 1
  - `solver`: `liblinear`
- **SVM**:
  - `C`: 10
  - `kernel`: `rbf`

---

## **Usage Instructions**

### **Prerequisites**
- Python 3.10 or higher.
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `umap-learn`.

### **Setup**
1. Clone the repository:
   ```bash
   https://github.com/pablolozanoa/DPaA-Project---Group-2.git
   cd DPaA-Project---Group-2


## Authors and Acknowledgment

This project was created by 
- [Nicolas Corsini](https://github.com/NCSanto01)
- [Pablo Lozano](https://github.com/pablolozanoa) 
- [Nicolas Rigau](https://github.com/nicorigau) 
- [Carlota Ruiz de Conejo](https://github.com/carlotardelasen) 
- [Raquel Vila](https://github.com/Raquelvilargz)