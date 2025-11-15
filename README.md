# ✈️ SkyScan: Decoding Airline Passenger Satisfaction with Machine Learning

## 📌 Overview

In the dynamic world of aviation, understanding what keeps passengers satisfied is critical to survival. This project explores a rich dataset of over 100,000 airline passenger surveys to identify what drives satisfaction and builds predictive models to classify passenger experiences.

By combining robust data cleaning, exploratory visualizations, feature engineering, and ensemble machine learning models, our solution predicts customer satisfaction with **93% accuracy** — offering strategic insights for airlines to optimize services.

---

## 🎯 Problem Statement

With fierce competition in the airline industry, enhancing passenger experience is a priority.  
**Objective**: Predict whether a passenger is "Satisfied" or "Neutral/Dissatisfied" using their demographic, service experience, and flight details.

---

## 👨‍💻 Team Members & Roles

| Name                 |
|----------------------|
| **Jaspinder Singh**   |
| **Divya Darshini. S** |
| **Pragun Lakhanpal**       |
| **Omkar Bhatawadekar**|
| **Avisha Jain**|
| **Saru Anoushka G V**|


---

## 🧰 Technologies & Tools

| Category            | Tools & Libraries                            |
|---------------------|-----------------------------------------------|
| Programming         | Python                                        |
| Data Handling       | Pandas, NumPy                                 |
| Visualization       | Matplotlib, Seaborn                           |
| Feature Scaling     | RobustScaler                                  |
| ML Models           | Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, AdaBoost |
| Model Evaluation    | ROC-AUC, Accuracy, Precision, Recall, F1-Score |
| Clustering          | K-Means                                       |
| Dimensionality Reduction | PCA                                     |
| Development Tools   | Jupyter Notebook, Google Colab, GitHub        |

---

## 🗂️ Project Structure

---

## 🧭 Step-by-Step Workflow

### 🔹 1. Data Acquisition
- Source: [Kaggle Airline Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- Size: 103,904 rows × 25 columns

### 🔹 2. Data Cleaning & Preprocessing
- Removed nulls and irrelevant columns (`id`, `Arrival Delay`)
- Addressed multicollinearity and scaled features using **RobustScaler**

### 🔹 3. Exploratory Data Analysis (EDA)
- Identified strong positive correlations between:
  - **Inflight Entertainment** & Satisfaction
  - **Type of Travel** (Business > Personal) & Satisfaction

### 🔹 4. Feature Engineering
- Created **“Delay Status”**: Categorized delay patterns
- Applied PCA to reduce dimensionality (18 ➝ 9 principal components)

### 🔹 5. Model Building
- Evaluated 8 ML models
- Applied K-Fold Cross-Validation
- Hyperparameter tuning via GridSearchCV

### 🔹 6. Model Evaluation

| Model         | Accuracy | ROC-AUC |
|---------------|----------|---------|
| Logistic Reg  | 87.32%   | 0.93    |
| KNN           | 91.57%   | 0.97    |
| **XGBoost**   | **93.39%** | **0.98** |
| Random Forest | 91.40%   | 0.97    |

🏆 **Best Model**: XGBoost — achieved top performance with minimal need for tuning.

---

## 📊 Key Insights

- **Inflight Entertainment, Online Boarding, and Seat Comfort** are top satisfaction drivers.
- **Business travelers** show significantly higher satisfaction.
- **Delays (arrival + departure)** strongly affect dissatisfaction.
- Minimal gains from hyperparameter tuning → models are near optimal.

---

## 💡 Recommendations

1. **Upgrade Inflight Entertainment** to retain premium travelers.
2. **Optimize Online Booking** to enhance convenience.
3. **Prioritize Delay Communication** especially for personal travelers.
4. **Segment Marketing Efforts** by travel type and delay experience.

---

## 🚀 Future Work

- Deploy model as a **real-time web app**
- Perform deeper **PCA component interpretation**
- Integrate **deep learning models** for further accuracy boosts

---

## 🙌 Acknowledgements

Thanks to **Great Learning** and our mentor **Mr. Animesh Tiwari** for guidance.

---

## 📫 Contact

For inquiries or collaborations, reach out via GitHub or [LinkedIn](https://www.linkedin.com/).

---


