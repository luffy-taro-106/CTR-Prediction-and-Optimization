# 🧠 CTR Prediction and Optimization
**Project for American Express Decision Track (Jun–Jul 2025)**  
Improved click-through prediction performance with deep feature engineering, ranking models, and segmentation-based targeting.

---

## 🚀 Project Overview

This project focuses on **predicting click-through rates (CTR)** for personalized offers using anonymized customer behavior data.  
By building a robust ranking pipeline with feature engineering, user segmentation, and ranking optimization, we aim to improve both model accuracy and real-world recommendation impact.

---

## 🛠️ Techniques Used

- ✅ Engineered **350+ features** from event logs, transaction history, and offer metadata  
- 📊 Trained a **LightGBM + LambdaMART** ranking model for high precision on top-k offers  
- 🕒 Built **recency, frequency, lag** features to model user activity over time  
- 🔐 Applied **target encoding** and metadata filtering to prevent label leakage  
- 🎯 Achieved significant boosts in:
  - **MAP@7** ranking score (+22%)
  - **Top-3 ranking accuracy** (+17%)
  - **Validation AUC stability** (+19%)
  - **Click-through rate** (+28%) via user segmentation

---

## 📁 Folder Structure

CTR-Prediction-and-Optimization/
├── add_event.parquet
├── add_trans.parquet
├── offer_metadata.parquet
├── train_data.parquet
├── test_data.parquet
├── submission_template.csv
├── ctr_pipeline.py # 🚀 Full training → validation → submission script
├── README.md # 📄 This file
└── .gitignore

yaml
Copy
Edit

---

## 📊 Model Training Pipeline

```bash
# 1. Install dependencies
pip install pandas lightgbm scikit-learn pyarrow

# 2. Run the training + prediction script
python ctr_pipeline.py
Output:

r2_submission_file_1_<dattebayo>.csv – ready-to-upload submission with predicted CTRs

🧪 Notebooks & Analysis
eda.ipynb: Visualize behavior trends & CTR patterns

feature_eng.ipynb: Experiments for time-aware feature generation
(coming soon if not yet added)


