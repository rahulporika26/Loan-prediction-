
# Loan Status Prediction

This project predicts whether a loan application will be **approved** or **rejected** based on applicant details using machine learning techniques.

## 📌 Project Overview

Financial institutions face challenges in identifying applicants who are likely to default. By analyzing applicant data (income, loan amount, credit history, etc.), this project builds a machine learning model to predict **Loan Status**.

## ⚙️ Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Training and testing ML models
* Final prediction system

## 📂 Project Structure

```
Loan_Status_Prediction.ipynb   # Main Jupyter Notebook
README.md                      # Project documentation
```

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn (SVC model)
* Jupyter Notebook

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone <your-repo-link>
   cd Loan-Status-Prediction
   ```
2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook Loan_Status_Prediction.ipynb
   ```

## 📊 Model Implemented

* **Support Vector Classifier (SVC)** with a linear kernel

  * Parameters:

    ```python
    SVC(C=1.0, kernel='linear', degree=3, gamma='scale')
    ```
* Training Accuracy: **79.86%**
* Test Accuracy: **83.33%**

## ✅ Results

The **SVC (linear kernel)** model generalized well, achieving higher accuracy on test data compared to training data.
This indicates the model is neither overfitting nor underfitting.

## 📌 Future Improvements

* Hyperparameter tuning (C, kernel, gamma)
* Try ensemble models (Random Forest, XGBoost, LightGBM)
* Deploy the model as a web app for end-users

## 👤 Author

**Rahul Naik Porika**

