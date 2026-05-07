# 🎓 Student Performance Prediction

A machine learning project that predicts students' final grades (G3) using 5 regression models, with a full interactive web app built with Streamlit.

🔗 **Live App:** [student-performance-project-qfpr33dmljmxwu8ntmgu5u.streamlit.app](https://student-performance-project-qfpr33dmljmxwu8ntmgu5u.streamlit.app/)

---

## 📁 Project Structure

```
├── Student_Performance_Project_ML.ipynb   # Full training notebook
├── app.py                                 # Streamlit web app
├── requirements.txt                       # Dependencies
└── student_data.csv                       # Dataset
```

---

## 📊 Dataset

The dataset used is the **Student Performance Dataset** collected from two Portuguese secondary schools.

| Property | Value |
|----------|-------|
| Entries | 395 students |
| Columns | 33 columns |
| Target | G3 (Final Grade, 0–20) |
| Problem Type | **Regression** |

### Column Names

| Category | Columns |
|----------|---------|
| Grades | G1, G2, G3 |
| Demographics | school, sex, age, address, famsize, Pstatus |
| Family | Medu, Fedu, Mjob, Fjob, guardian, famrel, famsup |
| Academic | studytime, failures, schoolsup, paid, activities, higher, nursery, reason |
| Lifestyle | internet, romantic, freetime, goout, Dalc, Walc, health, traveltime, absences |

---

## 🔄 Project Pipeline

### 1. 📦 Import Libraries
Imported all required libraries — pandas, numpy, matplotlib, seaborn, scikit-learn.

### 2. 📂 Load Data
Loaded `student_data.csv` and displayed basic info using `df.head()`, `df.info()`, and `df.describe()`.

### 3. 🔧 Preprocessing
- Used **LabelEncoder** to convert all categorical columns to numeric values.
- Checked for null values — dataset has no missing values.

### 4. 🔍 EDA (Exploratory Data Analysis)
- **G3 Distribution** — histogram and boxplot to understand the target variable.
- **Class Balance Check** — categorized G3 into Fail / Pass / Good / Excellent to visualize distribution.
- **Grade Distributions** — compared G1, G2, and G3 distributions.
- **Failures vs G3** — students with more failures tend to get lower grades.
- **Study Time vs G3** — more study time correlates with higher grades.
- **Absences vs G3** — scatter plot with trend line showing negative correlation.
- **Higher Education vs G3** — students aiming for higher education perform better.
- **Alcohol Consumption vs G3** — both weekday and weekend alcohol negatively affect grades.
- **Pairplot** — relationships between G1, G2, G3, failures, absences, studytime.
- **Correlation Heatmap** — full feature correlation matrix.

### 5. 🧹 Outlier Detection & Treatment
- Visualized outliers using boxplots for: age, absences, G1, G2, G3.
- Applied **IQR Capping** on `absences` — the only feature with significant outliers.

### 6. 🎯 Feature Selection
Dropped features with no direct impact on the final grade:
`school`, `reason`, `guardian`, `nursery`, `Pstatus`, `sex`, `age`

Remaining: **24 features** used for training.

### 7. ✂️ Data Splitting & Scaling
- Split: **80% train / 20% test** with `random_state=42`.
- Applied **StandardScaler** on all linear models (ElasticNet, Ridge, Lasso, SVR).
- Random Forest uses raw (unscaled) data.

### 8. 🤖 Model Training

| Model | Why we used it |
|-------|---------------|
| **ElasticNet** | Combines L1 + L2 regularization — handles correlated features and performs automatic feature selection |
| **Ridge** | L2 regularization — shrinks all coefficients, prevents overfitting in linear models |
| **Lasso** | L1 regularization — zeroes out irrelevant features, acts as built-in feature selection |
| **SVR** | Support Vector Regressor with linear kernel — effective on scaled data with clear linear patterns |
| **Random Forest** | Ensemble of 200 decision trees — captures non-linear relationships and feature interactions |

### 9. 🔍 Overfitting Check
Compared Train RMSE vs Test RMSE for each model. A difference below 0.5 is considered acceptable.

### 10. 📊 Model Comparison
Evaluated all models using **RMSE**, **MAE**, and **R²**.

### 11. 🏆 Select Best Model
Automatically selected the model with the lowest Test RMSE.

### 12. 🔮 Predictions
Generated actual vs predicted values for the best model with 3 visualizations:
- Scatter plot coloured by error magnitude
- Bar chart comparing actual vs predicted per student
- Residual plot

---

## 📈 Model Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| ElasticNet | 2.063 | 1.202 | 0.792 |
| Ridge | 2.235 | 1.526 | 0.756 |
| Lasso | 2.063 | 1.202 | 0.792 |
| SVR | 2.107 | 1.252 | 0.784 |
| **Random Forest** ⭐ | **1.781** | **1.120** | **0.845** |

> **Best Model: Random Forest** — lowest RMSE and highest R²

---

## 🔬 Sample Predictions — Random Forest

| # | Actual G3 | Predicted G3 | Difference |
|---|-----------|--------------|------------|
| 1 | 10 | 8.2 | -1.8 |
| 2 | 12 | 12.0 | 0.0 |
| 3 | 5 | 4.9 | -0.1 |
| 4 | 10 | 9.0 | -1.0 |
| 5 | 9 | 8.9 | -0.1 |
| 6 | 13 | 12.8 | -0.2 |
| 7 | 18 | 17.1 | -0.9 |
| 8 | 6 | 7.4 | +1.4 |
| 9 | 14 | 12.7 | -1.3 |
| 10 | 15 | 15.1 | +0.1 |

---

## 🌐 Web App — Streamlit

Built a full interactive web app with 6 pages:

| Page | Description |
|------|-------------|
| 🏠 Overview | Dataset summary, G3 distribution, class balance, model descriptions |
| 📊 Model Comparison | RMSE / MAE / R² comparison table and charts |
| 🔍 Overfitting Check | Train vs Test RMSE for all models |
| 🔮 Predict Grade | Input student features and get instant predictions from all 5 models |
| 📈 Feature Importance | Random Forest feature importance visualization |
| 🏆 Best Model Results | Scatter plot, bar chart, residual plot, and full predictions table |

🔗 **Live Demo:** [student-performance-project-qfpr33dmljmxwu8ntmgu5u.streamlit.app](https://student-performance-project-qfpr33dmljmxwu8ntmgu5u.streamlit.app/)

---
