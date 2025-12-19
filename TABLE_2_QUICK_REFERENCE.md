# 🎯 Quick Reference: Updated Table 2 (All 7 Models)

## Copy-Paste Ready Table for Dissertation

### Table 2: Cross-Validation Results (All 7 Models)

| Model | Mean Accuracy | SD | Min | Max |
|-------|---------------|-----|-----|-----|
| Logistic Regression | 0.9647 | 0.0321 | 0.9412 | 1.0000 |
| Random Forest | 0.9706 | 0.0274 | 0.9412 | 1.0000 |
| SVM | 0.9588 | 0.0401 | 0.8824 | 1.0000 |
| Gradient Boosting | 0.9647 | 0.0321 | 0.9412 | 1.0000 |
| XGBoost | 0.9706 | 0.0274 | 0.9412 | 1.0000 |
| LightGBM | **0.9765** | 0.0210 | 0.9412 | 1.0000 |
| **Deep Learning (MLP)** | 0.9588 | 0.0494 | 0.8824 | 1.0000 |

---

## What Changed

✅ **Added:** Deep Learning (MLP) - 7th model now included  
✅ **Format:** More readable with SD instead of ± notation  
✅ **Highlight:** LightGBM is best (0.9765, lowest SD)  

---

## Ready-to-Copy Interpretation

**Short version (2-3 sentences):**
```
All seven models achieved >95% cross-validated accuracy (Table 2), 
indicating robust generalization. Low standard deviation (SD ≤ 0.05) 
across folds suggests minimal overfitting. LightGBM demonstrated the 
highest and most stable performance (97.65% ± 2.1%), while the Deep 
Learning MLP showed comparable accuracy with higher variance (95.88% ± 
4.9%), indicating sensitivity to training data composition.
```

**Long version (with statistical limitation statement):**
```
Cross-Validation Results

All seven machine learning models were evaluated using stratified 
5-fold cross-validation to assess generalization performance. Results 
are presented in Table 2. All models achieved >95% mean cross-validated 
accuracy, with LightGBM demonstrating the highest performance (97.65% ± 
2.1%). The consistency across folds (low standard deviation, SD ≤ 5%) 
indicates minimal overfitting and suggests models generalize well to 
unseen data.

The Deep Learning MLP achieved 95.88% accuracy with higher variance 
(SD = 4.9%) compared to tree-based methods, suggesting greater 
sensitivity to the composition of training folds. Formal statistical 
comparison between models was not performed due to multiple models 
achieving 100% test accuracy and the small size of the IGD-positive 
class (n=16 total, n=3 per test fold), both of which violate 
assumptions of parametric hypothesis testing.
```

---

## Key Statistics Explained

| Metric | Your Value | What It Means |
|--------|-----------|--------------|
| **Mean Accuracy** | 0.9765 (LightGBM) | On average, model correct 97.65% of the time |
| **Standard Deviation** | 0.0210 | Variance across 5 folds is very small (±2.1%) |
| **Min** | 0.9412 | Worst fold achieved 94.12% accuracy |
| **Max** | 1.0000 | Best fold achieved 100% accuracy |
| **Low SD** | ≤ 0.05 | Stable model - doesn't overfit to specific folds |

---

## Where This Goes in Your Dissertation

**Section:** Results → Model Performance → Cross-Validation

**Location:** After Methods, before Test Set Results

**Before Test Set Results because:**
- CV provides more conservative, realistic estimate
- Less subject to small sample variability
- Appropriate for small positive class (n=16)

---

## What NOT to Do

❌ Don't say "Models were compared with t-tests"  
❌ Don't claim "statistical significance" without t-test values  
❌ Don't use only test accuracy (100% is not meaningful with n=3 positives)  
✅ DO use CV results (Table 2) as your primary performance metric  
✅ DO mention ROC curves for probability-based comparison  
✅ DO highlight that formal testing inappropriate due to sample size  

---

## Follow-Up Guidance

**If reviewer asks:** "Why no statistical tests comparing models?"

**Answer:** "Due to perfect accuracy achieved by multiple models on the 
test set (100%) and small positive class size (n=3), traditional 
hypothesis testing lacks validity. Cross-validation results (Table 2) 
provide more robust generalization estimates, while ROC and precision-
recall curves enable probability-based model comparison appropriate for 
imbalanced data."

---

## Quick Fact Check

Before submitting, verify:

- [ ] Table 2 includes all 7 models
- [ ] LightGBM values are 0.9765 ± 0.0210
- [ ] MLP is included with SD = 0.0494
- [ ] All SD values are ≤ 0.05 (indicating stability)
- [ ] Min/Max columns make sense (Min should be lowest, Max highest)
- [ ] No decimal rounding errors (e.g., 0.9412 not 0.941)

---

*Generated: 2025-12-19*  
*Updated to include all 7 models*  
*Ready to copy into dissertation*
