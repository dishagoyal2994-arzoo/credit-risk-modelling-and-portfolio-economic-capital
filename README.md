# Explainable Credit Risk Model: PD, Economic Capital & Compliance Reason Codes

Built on LendingClub's accepted-loans dataset (2007-2018) to mirror the full lifecycle
of a real credit risk model: default prediction, portfolio-level Economic Capital, and
model interpretability — the kind of workflow a bank's model development and model
validation teams both touch.

## Key Results
- **PD model:** XGBoost, ROC-AUC 0.708; threshold optimization lifted F1 from 0.125
  (default 0.5 cutoff) to 0.424
- **Multicollinearity check:** iterative VIF elimination caught severe collinearity
  between `grade`, `sub_grade`, and `int_rate` (VIF up to 35.5); dropped features
  until all VIF < 10
- **LGD, estimated from data, not assumed:** 92.4% realized LGD from 28,587 actual
  charge-offs — more than double the commonly-assumed Basel II 45% benchmark
- **Portfolio Economic Capital:** 50,000-path Vasicek single-factor Monte Carlo on a
  $518M test portfolio → Expected Loss $110M, 99.9% VaR $304M, Expected Shortfall
  $314M, Economic Capital $194M
- **Stress test:** Economic Capital nearly triples ($103M → $273M) as asset
  correlation moves from 0.05 to 0.30, while Expected Loss stays flat — correlation,
  not individual borrower risk, drives tail losses

## What's in the Notebook
1. Data cleaning and target construction (resolved loans only, explicit bias stated)
2. Feature selection via Information Value / Weight-of-Evidence, fit on train only
3. Multicollinearity diagnostics (correlation matrix + iterative VIF elimination)
4. Class imbalance handling (SMOTE, post-split only — avoids leakage)
5. Model comparison: Logistic Regression (WoE), Random Forest, XGBoost
6. Probability calibration and business-driven threshold selection
7. Portfolio Economic Capital: data-derived LGD/EAD, Vasicek Monte Carlo, VaR/ES,
   correlation stress testing
8. SHAP-based global and local interpretability, with a sanity check against
   domain intuition
9. A reason-code generator that translates SHAP outputs into ECOA/Regulation
   B-style adverse-action language — rule-based by default in this repo; designed
   to route through an LLM for more natural phrasing, with an ambiguity-based
   guardrail that flags low-confidence cases for manual review instead of forcing
   an explanation

## Key Methodological Choices
- Train/test split happens **before** any WoE fitting, IV screening, or SMOTE, to
  avoid statistics from the test set leaking into feature engineering
- Post-outcome columns (`recoveries`, `total_pymnt`, etc.) are explicitly excluded
  from the PD model and used only for LGD estimation
- `grade`/`sub_grade` are flagged as legitimate signal but discussed explicitly as
  "recovering LendingClub's own underwriting" rather than a from-scratch score

## Assumptions & Limitations
- Labels are defined only for resolved loans, which biases the observed default
  rate upward relative to the full portfolio
- LGD is a single portfolio-average estimate, not borrower-specific
- Asset correlation (ρ) in the Vasicek model is assumed, though its impact is
  explicitly stress-tested
- The analysis is based on a single lending platform's historical data and would
  need revalidation before use elsewhere

## Tech Stack
Python · pandas · scikit-learn · XGBoost · scorecardpy · imbalanced-learn · SHAP ·
statsmodels · matplotlib

## How to Run
Data: [LendingClub accepted loans, Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
(not included in this repo — download and place as `lendingclub_trimmed.csv` in the
working directory, or adjust the read path). Install dependencies, then run top to
bottom.
