# 🤖 Machine Learning with Python

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-11557C.svg)](https://matplotlib.org/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black.svg?logo=github&logoColor=white)](https://github.com/Satya-Siba-Nayak/MachineLearning-Python)

A collection of hands-on Machine Learning practicals, Exploratory Data Analysis (EDA) workflows, and regression & classification models implemented in Python using **Jupyter Notebooks**, **Pandas**, **NumPy**, **Matplotlib**, and **Scikit-Learn**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Notebooks Index & Quick Launch](#-notebooks-index--quick-launch)
- [Detailed Module Walkthrough](#-detailed-module-walkthrough)
  - [1. IRIS Dataset — Exploratory Data Analysis](#1-iris-dataset--exploratory-data-analysis)
  - [2. Titanic Dataset — EDA & Missing Value Imputation](#2-titanic-dataset--eda--missing-value-imputation)
  - [3. Binary Classification with Logistic Regression](#3-binary-classification-with-logistic-regression)
  - [4. Titanic Passenger Survival Prediction](#4-titanic-passenger-survival-prediction)
  - [5. Medical Insurance Charge Prediction — Linear vs. Polynomial Regression](#5-medical-insurance-charge-prediction--linear-vs-polynomial-regression)
- [Tech Stack & Dependencies](#-tech-stack--dependencies)
- [Getting Started & Local Setup](#-getting-started--local-setup)
- [Dataset Notes](#-dataset-notes)
- [Author & Acknowledgements](#-author--acknowledgements)

---

## 🔍 Overview

This repository demonstrates foundational to intermediate workflows in applied data science and machine learning:
- **Exploratory Data Analysis (EDA):** Inspecting shapes, distributions, data types, statistical metrics, and handling missing data using mean, median, and mode imputation strategies.
- **Data Preprocessing & Feature Engineering:** Handling missing values, one-hot/label mapping, polynomial feature transformations, and train-test partitioning.
- **Supervised Regression:** Simple Linear Regression and multi-degree Polynomial Regression ($d = 2, 3, 4, 5$) evaluated via $R^2$, MAE, MSE, and RMSE.
- **Supervised Classification:** Binary Logistic Regression models evaluated using Accuracy, Precision, Recall, F1-Score, and prediction probability outputs (`predict_proba`).

---

## 📂 Repository Structure

```text
MachineLearning-Python/
│
├── 📁 Binary-Classification-Logistic-Regression/
│   └── 📓 Logistic_Regression.ipynb         # Foundational binary logistic regression workflow
│
├── 📁 Insurance-Charge/
│   └── 📓 Insurance_charge_prediction.ipynb # Linear & Polynomial regression (Degrees 2-5)
│
├── 📁 IRIS-EDA/
│   └── 📓 Practical_1_Satya.ipynb           # 10 core EDA practical tasks on the Iris dataset
│
├── 📁 Titanic-EDA/
│   └── 📓 Titanic-EDA.ipynb                 # Titanic dataset inspection & missing value imputation
│
├── 📁 Titanic-Logistic_Regression/
│   └── 📓 Titanic_Logistic_Regression.ipynb # Logistic regression to predict Titanic passenger survival
│
└── 📄 README.md                             # Project documentation & navigation hub
```

---

## 🚀 Notebooks Index & Quick Launch

Click on any notebook name to navigate to the file, or use the **Open in Colab** badge to launch and execute the code in Google Colab immediately:

| Module / Project | Notebook File | Focus Area | Algorithm / Techniques | Colab |
| :--- | :--- | :--- | :--- | :--- |
| **Iris EDA** | [`Practical_1_Satya.ipynb`](./IRIS-EDA/Practical_1_Satya.ipynb) | Exploratory Data Analysis | Pandas queries, Sampling, Descriptive Stats, Column Dropping | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Satya-Siba-Nayak/MachineLearning-Python/blob/main/IRIS-EDA/Practical_1_Satya.ipynb) |
| **Titanic EDA** | [`Titanic-EDA.ipynb`](./Titanic-EDA/Titanic-EDA.ipynb) | Data Cleaning & Imputation | `dropna`, Mean/Median/Mode imputation, Feature/Target splitting | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Satya-Siba-Nayak/MachineLearning-Python/blob/main/Titanic-EDA/Titanic-EDA.ipynb) |
| **Binary Classification** | [`Logistic_Regression.ipynb`](./Binary-Classification-Logistic-Regression/Logistic_Regression.ipynb) | Binary Classification | Train/Test split, `LogisticRegression`, Accuracy, Probability estimation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Satya-Siba-Nayak/MachineLearning-Python/blob/main/Binary-Classification-Logistic-Regression/Logistic_Regression.ipynb) |
| **Titanic Survival** | [`Titanic_Logistic_Regression.ipynb`](./Titanic-Logistic_Regression/Titanic_Logistic_Regression.ipynb) | Applied Classification | Categorical encoding, Median fill, Logistic Regression, Classification Report | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Satya-Siba-Nayak/MachineLearning-Python/blob/main/Titanic-Logistic_Regression/Titanic_Logistic_Regression.ipynb) |
| **Insurance Charge Prediction** | [`Insurance_charge_prediction.ipynb`](./Insurance-Charge/Insurance_charge_prediction.ipynb) | Regression Analysis | Linear Regression, `PolynomialFeatures` ($d=2\dots5$), Curve Fitting, Evaluation ($R^2$, MAE, RMSE) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Satya-Siba-Nayak/MachineLearning-Python/blob/main/Insurance-Charge/Insurance_charge_prediction.ipynb) |

---

## 🔬 Detailed Module Walkthrough

### 1. IRIS Dataset — Exploratory Data Analysis
- **Folder:** [`IRIS-EDA/`](./IRIS-EDA/)
- **Notebook:** [`Practical_1_Satya.ipynb`](./IRIS-EDA/Practical_1_Satya.ipynb)
- **Dataset:** Iris Flower Dataset (`Iris.csv`)
- **Key Concepts & Exercises:**
  - Loading datasets and previewing records (`head()`, `sample()`).
  - Dimensionality check (`shape`) and column schemas (`columns`, `dtypes`).
  - Checking for missing values with `isnull().sum()`.
  - Analyzing target class distributions via `unique()` and `value_counts()`.
  - Computing summary statistics: mean, median, and standard deviation for features like `SepalWidthCm`.
  - Dataframe transformations and feature removal (`df.drop("Id", axis=1)`).

---

### 2. Titanic Dataset — EDA & Missing Value Imputation
- **Folder:** [`Titanic-EDA/`](./Titanic-EDA/)
- **Notebook:** [`Titanic-EDA.ipynb`](./Titanic-EDA/Titanic-EDA.ipynb)
- **Dataset:** Titanic Train Dataset (`train.csv`)
- **Key Concepts & Exercises:**
  - Initial dataset inspection with `head()`, `tail()`, `info()`, and `describe()`.
  - Comprehensive missing data analysis on key columns (`Age`, `Cabin`, `Embarked`).
  - **Comparative Imputation Strategies:**
    1. Complete case analysis via row removal (`dropna()`).
    2. Numerical imputation using Mean (`Age.fillna(mean)`).
    3. Numerical imputation using Median (`Age.fillna(median)`).
    4. Categorical imputation using Mode (`Embarked.fillna(mode()[0])`).
  - Segregation of independent feature variables ($X$) and target dependent variable ($Y = \text{Survived}$).
  - Identification and filtering of numerical vs categorical attributes.

---

### 3. Binary Classification with Logistic Regression
- **Folder:** [`Binary-Classification-Logistic-Regression/`](./Binary-Classification-Logistic-Regression/)
- **Notebook:** [`Logistic_Regression.ipynb`](./Binary-Classification-Logistic-Regression/Logistic_Regression.ipynb)
- **Key Concepts & Exercises:**
  - Introduction to the classification pipeline in `scikit-learn`.
  - Synthetic input features partitioned with an 80/20 train-test split (`test_size=0.2, random_state=42`).
  - Fitting `LogisticRegression()` and computing test set predictions.
  - Evaluation of model performance using `accuracy_score`.
  - Custom sample inference: predicting both hard class labels (`predict`) and class membership probabilities (`predict_proba`).

---

### 4. Titanic Passenger Survival Prediction
- **Folder:** [`Titanic-Logistic_Regression/`](./Titanic-Logistic_Regression/)
- **Notebook:** [`Titanic_Logistic_Regression.ipynb`](./Titanic-Logistic_Regression/Titanic_Logistic_Regression.ipynb)
- **Dataset:** Titanic Disaster Passenger Records (`Titanic-Dataset.csv`)
- **Key Concepts & Exercises:**
  - **Feature Selection:** Selecting `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, and `Fare`.
  - **Data Cleaning:** Imputing missing `Age` and `Fare` using median values.
  - **Encoding:** Binary transformation of categorical `Sex` (`female: 1, male: 0`).
  - **Model Training:** Training `LogisticRegression(max_iter=1000)` on an 80/20 train-test split.
  - **Evaluation:** Detailed evaluation with `accuracy_score` and `classification_report` (Precision, Recall, F1-Score).
  - **Inference Simulation:** Predicting survival probability for custom passenger profiles (e.g., 1st Class, Female, 28 years old).

---

### 5. Medical Insurance Charge Prediction — Linear vs. Polynomial Regression
- **Folder:** [`Insurance-Charge/`](./Insurance-Charge/)
- **Notebook:** [`Insurance_charge_prediction.ipynb`](./Insurance-Charge/Insurance_charge_prediction.ipynb)
- **Dataset:** Medical Insurance Costs (`insurance.csv`)
- **Key Concepts & Exercises:**
  - Exploratory inspection and correlation visualization of `BMI` vs `charges`.
  - Baseline Simple Linear Regression modeling:
    - Evaluating slope (coefficient), intercept, and baseline $R^2$ score.
    - Regression line visualization plotted against actual data.
  - **Polynomial Regression Exploration ($d \in [2, 3, 4, 5]$):**
    - Feature transformation using `PolynomialFeatures(degree=d, include_bias=False)`.
    - Model comparison across degrees using **Training $R^2$**, **Testing $R^2$**, **MAE**, **MSE**, and **RMSE**.
    - Plotting smooth polynomial regression curves across the continuous BMI spectrum.
  - Predicting healthcare charges for new target BMI values across multiple polynomial degrees.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.8+
- **Environment:** Jupyter Notebook / JupyterLab / Google Colab
- **Libraries & Frameworks:**
  - [Pandas](https://pandas.pydata.org/) — Data manipulation, slicing, cleaning, and aggregation
  - [NumPy](https://numpy.org/) — Numerical operations, array transformations, and matrix operations
  - [Matplotlib](https://matplotlib.org/) — Scatter plots, regression lines, and polynomial curve visualization
  - [Scikit-Learn](https://scikit-learn.org/) — Model selection, polynomial features, linear regression, logistic regression, and evaluation metrics

---

## 💻 Getting Started & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Satya-Siba-Nayak/MachineLearning-Python.git
cd MachineLearning-Python
```

### 2. Set Up a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### 4. Launch Jupyter Notebook or JupyterLab
```bash
jupyter notebook
```
or
```bash
jupyter lab
```

Navigate to any `.ipynb` file in your browser to run the cells sequentially.

---

## 📊 Dataset Notes

When running notebooks locally, ensure the relevant CSV datasets are available in your working directory or adjust the `pd.read_csv(...)` paths accordingly:

- **Iris Dataset (`Iris.csv`):** Standard dataset containing sepal and petal measurements across three iris species.
- **Titanic Dataset (`train.csv` / `Titanic-Dataset.csv`):** Historical passenger manifest with survival records from the Titanic disaster.
- **Insurance Dataset (`insurance.csv`):** Medical cost dataset covering age, sex, BMI, children, smoker status, region, and annual insurance charges.

> [!TIP]
> If running in Google Colab, you can upload the dataset files directly to the `/content/` directory via the files pane on the left, or mount your Google Drive as demonstrated in the notebooks.

---

## 👤 Author & Acknowledgements

- **Author:** [Satya Siba Nayak](https://github.com/Satya-Siba-Nayak)
- **Repository:** [MachineLearning-Python](https://github.com/Satya-Siba-Nayak/MachineLearning-Python)

---
*Feel free to star ⭐ this repository if you find these practicals and notebooks helpful!*
