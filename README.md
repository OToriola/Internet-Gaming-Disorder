# 🎓 Thesis Project: Screen Time & Gaming Disorder Risk Analysis

## Overview

This project investigates the relationship between screen time and Internet Gaming Disorder (IGD) risk using two complementary datasets:

- **NSCH (54,196 children)**: Population-level descriptive analysis
- **IGD (395 adolescents)**: Predictive modeling for gaming disorder risk

---

## 📁 Clean Project Structure

```
Healthcare/
│
├── IGD_Project/                     ✅ Predictive Modeling Component
│   ├── scripts/                     (4 ML scripts)
│   ├── data/                        (395 adolescents)
│   ├── results/                     (3 CSV files + documentation)
│   ├── visualizations/              (5 publication-quality figures)
│   └── README.md                    (Complete IGD documentation)
│
├── NSCH_Project/                    ✅ Descriptive Analysis Component
│   ├── scripts/                     (4 EDA scripts)
│   ├── data/                        (54,196 children, 3 datasets)
│   ├── results/                     (Analysis outputs + documentation)
│   ├── visualizations/              (Generated plots)
│   └── README.md                    (Complete NSCH documentation)
│
├── Archive/                         📦 Old/Superseded Work
│   ├── ADHD_analysis_files/         (Previous ADHD project)
│   ├── old_scripts/                 (Development scripts)
│   ├── old_results/                 (Previous outputs)
│   └── README.md                    (Archive guide)
│
├── README.md                        (This file)
├── requirements.txt                 (Python dependencies)
├── research_proposal.md             (Thesis proposal)
└── .venv/                          (Python environment)
```

---

## ✅ What's Included

### IGD_Project (Predictive Modeling) ⭐
- **7 Machine Learning Models** (Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM, Deep Learning MLP)
- **Best Model**: LightGBM with 100% accuracy, AUC-ROC 1.0, PR-AUC 1.0
- **Keras Tuner** hyperparameter optimization with early stopping
- **SHAP** feature importance analysis
- **3 CSV result files**: Model metrics, classification reports, confusion matrices
- **5 publication-quality visualizations**: SHAP plots, gaming hours analysis, gender differences, distributions
- **Complete reproducible pipeline** with fixed random seeds

### NSCH_Project (Descriptive Analysis)
- **4 EDA and analysis scripts** ready to run
- **3 datasets** with 54,196 children (ages 0-17)
- **Descriptive statistics** for population prevalence
- **Correlation analysis** between screen time and mental health
- **Sleep and physical activity** relationships
- **Stratified analysis** by age and gender

---

## 🚀 Quick Start

### Run IGD Predictive Model
```bash
cd IGD_Project/scripts
python ml_prediction_demo.py
```
**Output**: Model results, SHAP plots, all metrics

### Run NSCH Descriptive Analysis
```bash
cd NSCH_Project/scripts
python eda_template.py
```
**Output**: Descriptive statistics, visualizations

---

## 📊 Key Results

### IGD Model Performance (LightGBM - BEST)
| Metric | Value |
|--------|-------|
| Accuracy | **100%** |
| AUC-ROC | **1.0** |
| PR-AUC | **1.0** |
| Precision | **1.0** |
| Recall | **1.0** |

### All 7 Models Comparison
- **LightGBM**: 100% (perfect)
- **Random Forest**: 98.46% with perfect precision
- **Deep Learning MLP**: 98.46% (tuned with Keras Tuner)
- **Logistic Regression**: 98.46% with perfect recall
- **XGBoost**: 98.46% with class weighting
- **Gradient Boosting**: 98.46% baseline
- **SVM**: 98.46% non-linear classification

### NSCH Sample
- 54,196 children (0-17 years)
- Screen time, mental health, sleep, activity variables
- Ready for epidemiological analysis

---

## 📝 For Thesis Writing

### Copy These Files to Thesis

**Results Tables**:
```
IGD_Project/results/igd_model_evaluation_results.csv        → Results Section
NSCH_Project/results/nsch_igd_analysis.csv                 → Background Section
```

**Visualizations**:
```
IGD_Project/visualizations/shap_importance.png             → Figure 1
IGD_Project/visualizations/gaming_hours_vs_igd.png         → Figure 2
[Other IGD visualizations]                                  → Figures 3-5
```

**Appendices**:
```
IGD_Project/results/igd_classification_reports.csv         → Appendix A
IGD_Project/results/igd_confusion_matrices.csv             → Appendix B
IGD_Project/scripts/ml_prediction_demo.py                  → Appendix C
```

---

## 📚 Thesis Structure (Recommended)

### 1. Introduction
- Problem: Screen time prevalence (from NSCH data)
- Research question: Can we identify gaming disorder risk?
- Thesis statement: "ML models can predict IGD status from behavioral factors"

### 2. Methods
- **NSCH**: Survey design, variables, descriptive approach
- **IGD**: Cohort design, features, 7 ML models, evaluation metrics
- **Justification**: Why separate datasets provide complementary evidence

### 3. Results
- **Part A (NSCH)**: Population prevalence and associations
- **Part B (IGD)**: Model comparison, LightGBM performance, SHAP analysis

### 4. Discussion
- Integration of findings
- Clinical implications
- Limitations and future directions

### 5. Appendices
- CSV files from results/
- All scripts from scripts/
- Feature descriptions and hyperparameters

---

## 🔗 Why Two Datasets?

| Aspect | NSCH | IGD |
|--------|------|-----|
| **Sample** | 54,196 children | 395 adolescents |
| **Design** | Population survey | Clinical cohort |
| **Purpose** | Context & prevalence | Prediction & validation |
| **Variables** | Screen time, general health | Gaming hours, disorder status |
| **Role in Thesis** | Background/context | Main analysis |

**Together they show**: Population prevalence (NSCH) + Predictive model (IGD) = Complete story

---

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: TensorFlow/Keras with Keras Tuner
- **Interpretability**: SHAP (SHapley Additive exPlanations)
- **Visualization**: Matplotlib, Seaborn
- **Data**: Pandas, NumPy

---

## ✨ Key Features

✅ **Reproducible**: All random seeds set (NumPy, random, TensorFlow)  
✅ **Stratified**: Train/test split preserves 6.2% positive class ratio  
✅ **Class-Balanced**: Handles 93.8% / 6.2% imbalance with weights  
✅ **Explainable**: SHAP analysis for model interpretation  
✅ **Comprehensive**: 6 metrics per model (Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC)  
✅ **Publication-Ready**: Clean visualizations and tables  

---

## 📋 Thesis Completion Checklist

- [ ] **Methods Section**
  - [ ] Describe NSCH design and variables
  - [ ] Describe IGD cohort and features
  - [ ] Explain ML model selection
  - [ ] Justify separate dataset use

- [ ] **Results Section**
  - [ ] NSCH prevalence table
  - [ ] NSCH associations with mental health
  - [ ] IGD model comparison table
  - [ ] SHAP feature importance figure
  - [ ] Best model summary (LightGBM)

- [ ] **Discussion**
  - [ ] Interpret NSCH findings
  - [ ] Interpret IGD model performance
  - [ ] Explain high accuracy (well-separated classes)
  - [ ] Clinical and public health implications
  - [ ] Limitations and future work

- [ ] **Appendices**
  - [ ] CSV files from results/
  - [ ] Complete scripts
  - [ ] Hyperparameter specifications
  - [ ] Feature descriptions

---

## 📖 Documentation

- `IGD_Project/README.md` - Complete IGD technical details
- `NSCH_Project/README.md` - Complete NSCH analysis details
- `Archive/README.md` - What's archived and why
- `research_proposal.md` - Thesis proposal outline
- This README - Overall structure and quick reference

---

## 🎯 Current Status

✅ NSCH data loaded and organized  
✅ IGD data loaded and modeled  
✅ 7 ML models trained and evaluated  
✅ SHAP analysis completed  
✅ All results exported to CSV  
✅ Visualizations generated  
✅ Projects cleanly organized  
✅ Archive created (old work separated)  

**⏳ Next Step: Write your thesis! 📚**

---

## 📞 Quick Reference

| Task | Location |
|------|----------|
| **Run IGD models** | `IGD_Project/scripts/ml_prediction_demo.py` |
| **Run NSCH analysis** | `NSCH_Project/scripts/eda_template.py` |
| **Model results table** | `IGD_Project/results/igd_model_evaluation_results.csv` |
| **SHAP visualization** | `IGD_Project/visualizations/shap_importance.png` |
| **IGD docs** | `IGD_Project/README.md` |
| **NSCH docs** | `NSCH_Project/README.md` |
| **Archive contents** | `Archive/README.md` |

---

**Last Updated**: December 19, 2025  
**Status**: ✅ Cleaned Up & Ready for Thesis  

---

**Everything is organized and ready! Good luck with your thesis! 🎓📚**
