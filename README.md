# ✈️ Airline Passenger Satisfaction — End-to-End ML Project

> Predicting whether a passenger will be satisfied with their flight and understanding what drives their experience.

---

## 📌 Project Overview

Airlines collect huge amounts of feedback about passenger experience — Wi-Fi, seat comfort, delays, check-in experience, and more.  

In this project, I built an **end-to-end machine learning pipeline** to:

1. **Predict whether a passenger is satisfied or dissatisfied** based on their journey and service ratings.  
2. **Identify the key drivers of satisfaction** (what actually makes people happy or unhappy).  
3. **Segment passengers into behavioral clusters** using unsupervised learning to help design targeted service improvements.

This project showcases my skills in:

- **Data cleaning & preprocessing**
- **Exploratory Data Analysis (EDA) & statistical testing**
- **Feature engineering**
- **Dimensionality reduction (PCA)**
- **Clustering (K-Means)**
- **Supervised learning (Logistic Regression, Tree-based models, XGBoost, etc.)**
- **Model evaluation (confusion matrix, ROC-AUC, cross-validation)**
- **Business-oriented interpretation of results**

---

## 🧾 Dataset

- **Source:** Airline passenger satisfaction survey (public dataset, e.g. Kaggle)
- **Rows:** 103,904 passengers  
- **Columns:** 25 (demographics, travel details, service ratings, delays, target label)

### 🔹 Key Features

- **Demographics:**  
  - `Gender`, `Age`
- **Travel Info:**  
  - `Customer_type` (Loyal / Disloyal), `Type_of_Travel` (Business / Personal)  
  - `Class` (Business / Eco / Eco Plus)  
  - `Flight_distance`
- **Service Ratings (1–5):**  
  - `Inflight_wifi_service`, `Online_boarding`, `Seat_comfort`, `Food_and_drink`,  
    `Inflight_entertainment`, `On-board_service`, `Leg_room_service`,  
    `Baggage_handling`, `Checkin_service`, `Inflight_service`, `Cleanliness`
- **Operational Metrics:**  
  - `Departure_delay_in_minutes`, `Arrival_delay_in_minutes`
- **Target Variable:**  
  - `satisfaction` → **Satisfied (1)** / **Neutral or Dissatisfied (0)**

---

## 🎯 Business Problem

Airlines want to:

- **Reduce dissatisfied passengers**
- **Understand which touchpoints matter most**
- **Prioritize investments (Wi-Fi, check-in, onboard service, etc.)**
- **Identify high-value passenger segments**

So the project answers:

1. _“Can we accurately predict passenger satisfaction?”_  
2. _“Which factors are most strongly associated with dissatisfaction?”_  
3. _“Can we group passengers into meaningful clusters based on their behavior and experience?”_

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Data Handling:** `pandas`, `numpy`  
- **Visualization:** `matplotlib`, `seaborn`  
- **Stats & ML:** `scikit-learn`, `xgboost`, `scipy`  
- **Preprocessing:** `RobustScaler`, one-hot encoding, PCA  
- **Clustering:** K-Means  
- **Model Evaluation:** Confusion matrix, Accuracy, Precision, Recall, ROC-AUC, K-Fold CV

---

## 🔍 Approach

### 1️⃣ Data Cleaning & Preparation

- Renamed columns for consistency and readability (e.g. spaces → underscores).
- Verified:
  - No duplicate `id` values.
  - Only **one column with missing values**: `Arrival_delay_in_minutes` (310 missing out of 103,904).
- Decided to:
  - Use delay fields for **feature engineering**.
  - Drop `Arrival_delay_in_minutes` later after extracting meaningful information.
- Dropped purely identifier columns (`id`, etc.) that do not contribute to prediction.

---

### 2️⃣ Exploratory Data Analysis (EDA)

Key steps:

- Checked class balance using a **pie chart** of `satisfaction`.  
  ➜ Mild class imbalance, but both satisfied and dissatisfied classes are well represented.
- Visualized:
  - Distribution of service ratings (1–5) across all rating variables.
  - Relationship between `Departure_delay_in_minutes` and `Arrival_delay_in_minutes` colored by `satisfaction`.
- Outlier analysis for delay columns using **IQR** and examined how outlier passengers are distributed across satisfaction levels.

🔎 **Statistical Tests Used**

To go beyond visuals, I used classical hypothesis testing:

- **Levene’s test** for equality of variances.
- **Two-sample t-tests** to compare means of continuous variables across satisfaction groups (e.g. Age, Flight distance, delay minutes).
- **Chi-square tests + Cramer’s V** to measure association strength between:
  - Categorical features (e.g. `Class`, `Customer_type`, `Type_of_Travel`, ratings)  
  - And the target `satisfaction`.

This helps quantify **which features are truly associated with satisfaction**, not just visually different.

---

### 3️⃣ Feature Engineering

✅ **Delay status feature**

Created a new categorical feature that summarizes both delay columns into a human-readable label:

- `No delay`
- `Arrival delay`
- `Departure delay`
- `Arrival and Departure delay`

```python
def delay_status(col):
    if col['Arrival_delay_in_minutes'] == 0 and col['Departure_delay_in_minutes'] == 0:
        return 'No delay'
    elif col['Arrival_delay_in_minutes'] != 0 and col['Departure_delay_in_minutes'] == 0:
        return 'Arrival delay'
    elif col['Arrival_delay_in_minutes'] == 0 and col['Departure_delay_in_minutes'] != 0:
        return 'Departure delay'
    else:
        return 'Arrival and Departure delay'
