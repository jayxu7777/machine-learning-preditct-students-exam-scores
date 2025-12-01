# machine-learning-preditct-students-exam-scores
python code to do machine learning
# Predicting Secondary School Student Performance

This project builds a complete machine learning pipeline in **Python** to predict secondary school students’ final performance (G3, score 0–20).  
The dataset includes demographic, lifestyle, family background, academic history, and school-related features.  
We evaluate multiple models under **three prediction tasks** and **three feature setups**.

📄 **Project Report:** See full PDF in the repository.

---

## 📌 Project Summary
This project applies supervised machine learning to predict three related targets:

1. **Binary classification** – Pass vs. Fail  
2. **Five-level classification** – Discretized academic performance (0–4)  
3. **Regression** – Predict continuous final grade G3

To test the contribution of early academic history, we define three feature setups:

- **Setup A** – All features including G1 & G2  
- **Setup B** – Includes G1 but removes G2  
- **Setup C** – Removes both G1 & G2

---

## 🧰 Methods and Pipeline

### **Data Preprocessing**
- One-hot encoding for categorical variables  
- Standardization (for SVM, MLP, LR)  
- Raw numeric inputs for tree-based models  
- Full preprocessing wrapped in `scikit-learn` Pipelines

### **Feature Selection**
- Embedded feature selection via a Random Forest  
- Keep top **K = 20** features per fold  
- Run separately in each CV fold to avoid leakage

### **Models Compared**
- Random Forest  
- XGBoost  
- SVM (RBF)  
- MLP neural network  
- Logistic Regression (classification)  
- Linear Regression / GLM (regression baseline)

Naïve baselines (G2, G1, or majority/mean) included for fair comparison.

### **Evaluation**
- Outer **90/10 train-test split**  
- Inner **repeated 10-fold CV** (20 fold-level scores)  
- Metrics:
  - Accuracy (PCC) for classification  
  - RMSE for regression  
- Paired t-tests vs. baselines  
- Confusion matrices and regression scatter plots on test set  

---

## 📊 Key Results

(**All results derived from your report** :contentReference[oaicite:1]{index=1})

### ✔ **Binary Classification**
- Setup A: **XGBoost ~92–93% PCC**, best overall  
- Setup B: RF and XGB around **90%**  
- Setup C: Best models still **mid-80%**, strong even without G1/G2  

### ✔ **Five-Level Classification**
- Setup A: RF reaches **73–74%** accuracy  
- Setup B: Best models around **57–58%**  
- Setup C: Models reach **≈37%**, outperforming earlier benchmarks  

### ✔ **Regression (RMSE)**
- Setup A: GLM & RF around **1.29–1.33** (excellent)  
- Setup B: RMSE **<1.90**  
- Setup C: RMSE **~2.70–2.85**, still structured, not random  

### ✔ **Feature Importance Insights**
Expected shifts appear:

- With G1/G2 → these dominate  
- Without G1/G2 → models rely on behavior, attendance, failures, family factors  
- Importance aligns across tasks and setups, showing model stability  

---

## 📂 Repository Structure (Recommended)

