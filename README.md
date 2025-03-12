# Machine Learning Models: Supervised and Unsupervised Learning

## Overview
This project implements both **Supervised Learning (SVM for Classification)** and **Unsupervised Learning (Gaussian Mixture Model for Clustering)** using a diabetes dataset. The goal is to compare how well each method performs in predicting diabetes.

## Dataset
The dataset used in both models is the **Pima Indians Diabetes Dataset**, which consists of medical diagnostic attributes to predict diabetes outcomes. It includes features like glucose level, BMI, age, and blood pressure.

## Supervised Learning: SVM Classifier
### Description
- Uses **Support Vector Machine (SVM)** with a linear kernel.
- Preprocesses the data with **StandardScaler**.
- Splits the data into **training (80%)** and **testing (20%)**.
- Trains an SVM model on the data.
- Evaluates performance using **accuracy score**.
- Implements a predictive system to classify a new patient as diabetic or non-diabetic.

### Dependencies
- `numpy`
- `pandas`
- `sklearn`

### How to Run
1. Ensure `diabetes.csv` is in the same directory.
2. Run the script:
   ```bash
   python supervised_ml.py
   ```
3. The model will print training and testing accuracy along with predictions.

---

## Unsupervised Learning: Gaussian Mixture Model (GMM) Clustering
### Description
- Uses **Gaussian Mixture Model (GMM)** for clustering.
- Applies **Principal Component Analysis (PCA)** to reduce dimensions while preserving 95% variance.
- Preprocesses the data with **StandardScaler**.
- Evaluates clustering performance using **Silhouette Score**.
- Determines the best covariance type for GMM.
- Maps cluster labels to actual classes and calculates accuracy.
- Generates a **PCA scatter plot with clustering results**.

### Dependencies
- `numpy`
- `pandas`
- `sklearn`
- `matplotlib`

### How to Run
1. Run the script:
   ```bash
   python unsupervised_ml.py
   ```
2. The script will output:
   - Best covariance type
   - Best silhouette score
   - Model accuracy
   - A saved **PCA scatter plot (cluster_plot.png)**

## Results
- **SVM Model (Supervised Learning):** Predicts diabetes with an accuracy score.
- **GMM Model (Unsupervised Learning):** Clusters patients into two groups and maps them to actual labels for evaluation.

## Conclusion
- **Supervised learning (SVM)** is more suitable for precise classification problems where labeled data is available.
- **Unsupervised learning (GMM)** is useful for discovering patterns in data but requires mapping labels to interpret results.

## Future Improvements
- Try other classification models like Random Forest or Neural Networks.
- Use more advanced clustering techniques like K-Means or DBSCAN.
- Tune hyperparameters for improved performance.

