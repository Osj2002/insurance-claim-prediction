# Predictive Analysis in Insurance Claims: Machine Learning & Customer Segmentation

## 📌 Project Overview
This repository contains a comprehensive data science project aimed at predicting health insurance claim behavior using statistical modeling, customer segmentation, and supervised machine learning classification algorithms. 

Traditional global predictive frameworks often struggle to classify minority, high-risk risk pools accurately. To address this gap, this project integrates an unsupervised **Hierarchical Clustering** pipeline using **Gower Distances** to segment policyholders into distinct risk profiles. Predictive benchmarks are then established across both global and cluster-specific models to optimize insurance risk management and strategic premium pricing.

---

## 📊 Data Description & Preprocessing
The analysis uses a structured health insurance dataset (sourced via Kaggle) containing **1,338 observations** and **8 attributes** mapping demographic and clinical characteristics of US beneficiaries:

### Feature Schema
* **`Age`** (Numerical): Age of the individual (ranges from 18 to 64 years; Mean: 39.21).
* **`Sex`** (Categorical): Gender (`Female` mapped to 0, `Male` mapped to 1).
* **`Bmi`** (Numerical): Body Mass Index ($kg/m^2$; Mean: 30.66, indicating an overweight average baseline).
* **`Children`** (Numerical): Number of dependents covered by the health insurance (ranges from 0 to 5).
* **`Smoker`** (Categorical): Smoking status (`Non-smoker` mapped to 0, `Smoker` mapped to 1).
* **`region`** (Categorical): US residential zone (`Northeast`: 0, `Northwest`: 1, `Southeast`: 2, `Southwest`: 3).
* **`charges`** (Numerical): Individual medical costs billed by the health insurance. *Note: Included as a predictor variable as costs are finalized prior to the eventual claim filing decision.*
* **`insuranceclaim`** (**Target Variable**): Operational outcome status indicating if an insurance claim was filed (`Yes`: 1, `No`: 0; 58.5% Claim rate vs 41.5% Non-claim rate).

### Preprocessing & Feature Engineering
1. **Data Cleaning:** Identified and dropped one duplicate record. Outliers present within medical charges were intentionally preserved to mirror real-world extreme medical events.
2. **Categorical Handling:** Applied Ordinal Encoding to binary structural pairs (`Sex`, `Smoker`) and One-Hot Encoding to the multi-class feature (`region`).
3. **Feature Scaling:** Standardized continuous variables (`Age`, `Bmi`, `Charges`, `Children`) utilizing a `StandardScaler` to bring all variations onto a standard Gaussian scale ($\mu=0, \sigma=1$).
4. **Data Splitting:** Maintained a strict structural split of 80% for training partitions and 20% for testing sets.

---

## 📈 Exploratory Data Analysis & Statistical Testing
* **Univariate Trajectories:** The medical `charges` distribution is significantly right-skewed, indicating the vast majority of policyholders incur bills under $20,000 USD.
* **Normality Verification:** A **Shapiro-Wilk Test** was conducted across all core numerical variables. Every feature returned $p < 0.05$, systematically rejecting the null hypothesis of normal distribution.
* **Non-Parametric Association:** Because normality assumptions were violated, a **Mann-Whitney U Test** was executed to evaluate distributional differences between claimants and non-claimants. The test confirmed that `Age`, `Bmi`, `Charges`, and `Children` all display statistically significant differences ($p < 0.05$) between the two groups.

---

## 🧩 Advanced Analysis: Mixed-Data Customer Segmentation

To parse overlapping sub-structures within mixed attributes, **Hierarchical Clustering** was performed paired with a calculated **Gower Distance Matrix**. **Silhouette Analysis** confirmed an optimal cluster topology of **2 clusters**.

### Cluster Profiles
* **Cluster 0: Low Risk Group (79% of Sample; $n = 1,064$)**
  * Consists entirely of **non-smokers**.
  * Characterized by significantly lower medical expenses (Mean Charge: $8,434.27 USD).
  * Displays a highly balanced internal claim behavior split (50.14% Claim vs 49.85% No-Claim).
* **Cluster 1: High Risk Group (21% of Sample; $n = 274$)**
  * Consists entirely of **smokers**.
  * Characterized by extreme medical liabilities (Mean Charge: $32,050.23 USD).
  * Heavily imbalanced towards insurance claims (90.87% Claim vs 9.12% No-Claim).

### Cluster Visualization
To evaluate spatial boundaries and validate cluster separation, **Multidimensional Scaling (MDS)** was implemented to project the multi-dimensional Gower matrix into a distinct 2D coordinate map, proving a sharp, isolated boundary between the two clusters.

---

## 🤖 Predictive Machine Learning Performance

All classification models were optimized using **5-Fold Cross-Validation** incorporating automated **Grid Search** for hyperparameter tuning. For the cluster-specific workflow, **SMOTE (Synthetic Minority Oversampling Technique)** was exclusively applied inside Cluster 1 training folds to remedy the severe 90.9% target imbalance.

### 1. Global Model Performance
The baseline global models evaluated on the unsegmented dataset produced the following performance metrics:

| Algorithm | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.8623 | 0.9172 | 0.8889 |
| Random Forest | 0.9551 | 0.9490 | 0.9521 |
| Support Vector Machine (SVM) | 0.9338 | 0.8981 | 0.9156 |
| **Decision Tree (Best Global)** | **0.9811** | **0.9936** | **0.9873** |

*Global Feature Importance:* The Global Decision Tree isolated `Bmi` and `Children` as the most critical predictors, while `Sex` and `Region` were found to be statistically insignificant.

### 2. Cluster-Specific Performance
Segmented data partitions produced localized models optimized for their respective risk profiles:

| Algorithm | Cluster 0 Precision | Cluster 0 Recall | Cluster 0 F1 | Cluster 1 Precision | Cluster 1 Recall | Cluster 1 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8571 | 0.8780 | 0.8675 | 0.9957 | 0.9317 | 0.9627 |
| **Random Forest** | 0.9572 | 0.9231 | 0.9398 | **0.9879** | **0.9839** | **0.9859** (Best C1) |
| SVM | 0.6178 | 0.5460 | 0.5797 | 0.9873 | 0.6265 | 0.7666 |
| **Decision Tree** | **0.9794** | **0.9831** | **0.9813** (Best C0) | 0.9839 | 0.9839 | 0.9839 |

* **Cluster 0 Optimal Model:** **Decision Tree** (97.9% Precision, 98.3% Recall). Core drivers: `Children`, `Age`, and `Bmi`.
* **Cluster 1 Optimal Model:** **Random Forest** post-SMOTE (98.8% Precision, 98.4% Recall). Core drivers: `Children` (highest importance), followed by `Bmi`, `Charges`, `Age`, and `Sex`.

---

## 🎯 Strategic Conclusion
While the standalone global Decision Tree retains excellent metrics (98.1% Precision), it runs the structural risk of misclassifying low-volume, high-risk segments in complex markets. **Cluster-specific modeling** mitigates this by applying targeted sub-routines (like SMOTE on Cluster 1) to isolate accurate predictive signals across both low-risk and high-risk consumer subsets, yielding an overall superior framework for live risk management and pricing validation.

---

## 🛠️ Execution & Requirements
To execute the pipelines locally, install the necessary computational dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn gower imbalanced-learn
