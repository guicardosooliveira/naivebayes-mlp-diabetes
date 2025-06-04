# Naivebayes/MLP Diabetes Prediction

This project applies supervised machine learning techniques to predict early-stage diabetes risk based on a set of medical and lifestyle-related features. It uses two classification models:

- **Naive Bayes**: used as a simple baseline.
- **Multi-Layer Perceptron (MLP)**: tuned using GridSearchCV to find the best combination of hyperparameters.

The dataset used comes from Kaggle: [Early Stage Diabetes Risk Prediction Dataset](https://www.kaggle.com/datasets/abdelazizsami/early-stage-diabetes-risk-prediction).

---

## 📁 Project Structure

```bash
naivebayes-mlp-diabetes/
│
├── data/
│ └── diabetes_data_upload.csv # Dataset (place it here)
│
├── notebook/
│ └── main.ipynb # Main notebook (run this)
│
├── src/
│ ├── preprocessing.py # Functions to load, clean and split the dataset
│ └── models.py # ML model training logic
│
├── env/ # (Optional) Virtual environment folder
├── README.md
└── .gitignore
```


## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/guicardosooliveira/naivebayes-mlp-diabetes
cd naivebayes-mlp-diabetes
```
### 2. Install dependencies

```bash
pip install pandas scikit-learn
```
### 3. Download the dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/abdelazizsami/early-stage-diabetes-risk-prediction

Place the file diabetes_data_upload.csv inside the data/ directory.

### 4. Run the main notebook

```bash
jupyter notebook notebook/main.ipynb
```

## How It Works
The data is loaded and preprocessed:

- Binary categorical features are converted to numeric.

- Gender and target class are encoded.

- The dataset is split into training and test sets.

Two models are trained:

- Naive Bayes using default parameters.

- MLP using GridSearchCV to find optimal hyperparameters.

Results are evaluated using the F1-Score and confusion matrix.

## Dependencies
- Python 3.7+

- pandas

- scikit-learn

- jupyter (for running notebooks)