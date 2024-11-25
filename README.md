# Breast Cancer Prediction using Random Forest Classifier

This project implements a machine learning model to predict whether a tumor is malignant or benign based on features derived from breast cancer biopsies. The model uses a **Random Forest Classifier** to classify the tumors, trained and evaluated using a dataset of breast cancer features.

## Project Overview

The purpose of this model is to provide an automated approach to classify breast cancer tumors. Early and accurate diagnosis of breast cancer is essential, and machine learning models like Random Forests can significantly help doctors in identifying malignant tumors.

The project involves the following steps:
1. **Data Preprocessing:** Loading and preparing the data for training by handling missing values, encoding categorical features, and scaling features.
2. **Model Training:** Training a Random Forest model to classify tumors based on the provided features.
3. **Evaluation:** Evaluating the model using various metrics like accuracy, precision, recall, F1-score, and ROC AUC to ensure its performance.
4. **Visualization:** Plotting key evaluation metrics like the ROC curve to visualize the model's ability to discriminate between malignant and benign tumors.

## Dataset

The dataset used for training the model comes from the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). This dataset contains 569 instances with 30 features such as radius, texture, smoothness, and others.

- **Class labels**: 
  - `0`: Benign
  - `1`: Malignant

You can either use the dataset directly (if included) or follow the instructions to load it.

## Setup Instructions

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/BreastCancerPrediction.git
   cd BreastCancerPrediction
2. Install the required dependencies:
   pip install -r requirements.txt
   Download the dataset (if you haven't already) from UCI Machine Learning Repository or use the one included in the data/ folder.

3. Run the Jupyter notebook:
   jupyter notebook breast_cancer_classifier.ipynb
   
## Model Details

The model was trained using Random Forest Classifier from scikit-learn:

- Random Forest Classifier is an ensemble learning method that combines the predictions of multiple decision trees.
- The model is evaluated using several performance metrics, including:
     - Accuracy: 97.08%
     - Precision: 96% for malignant class (1), 98% for benign class (0)
     - Recall: 99% for malignant class (1), 94% for benign class (0)
     - F1-Score: 97% weighted average
     - Confusion Matrix: To show the distribution of true positives, false positives, true negatives, and false negatives
     - ROC Curve: For evaluating the tradeoff between sensitivity and specificity.
       
## Evaluation Metrics

### Confusion Matrix
The confusion matrix is used to evaluate the performance of the classifier. The matrix for this model is:

- [[ 59   4]
- [ 1 107]]
  
#### Where:

 - True Negatives (TN): 59 (Benign correctly classified as benign)
 - False Positives (FP): 4 (Benign incorrectly classified as malignant)
 - False Negatives (FN): 1 (Malignant incorrectly classified as benign)
 - True Positives (TP): 107 (Malignant correctly classified as malignant)
   
### ROC Curve

The ROC curve is plotted to visualize the classifier's performance across different thresholds. A higher area under the curve (AUC) indicates better performance.

## Files

breast_cancer_classifier.ipynb: Jupyter notebook containing the full implementation of the model. The implementation also includes the loading of the dataset from sci-kit learn so a separate data file is not needed in this case.

## Future Improvements

 - Hyperparameter Tuning: Experimenting with different hyperparameters for the Random Forest model using grid search or random search to improve performance.
 - Other Algorithms: Trying different machine learning algorithms (e.g., SVM, XGBoost) and comparing their performance.
 - Cross-Validation: Implementing cross-validation for more reliable evaluation.
