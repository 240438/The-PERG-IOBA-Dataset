# Methodology 
The paper used 3 types of methodologies

## 1) Model 1 (Dual‑stream NN / feature fusion)

```text
                 ┌─────────────────────────────┐
PERG features ─▶│  PERG branch (MLP/Dense)    │──┐
(N35,P50,N95,    └─────────────────────────────┘  │
 amps, ratio,                                     │
 PTP, …)                                          │
                                                  ▼
                                           Concatenate
                                                  │
                                                  ▼
                                       ┌────────────────────┐
Clinical features ─▶┌────────────────▶│ Fusion Dense layers│──▶ Sigmoid ─▶ P(pathological)
(age, sex, VA)      │                  └────────────────────┘
                    │
                    ▼
          ┌─────────────────────────────┐
          │ Clinical branch (MLP/Dense) │
          └─────────────────────────────┘
```

    Key point: PERG and clinical features are learned separately, then merged inside one neural network before final classification.

## 2) Model 2 (Stacking ensemble / prediction fusion)

```text
                         (same input features to each base model)
                   ┌────────────────────────────────────────────────┐
Combined features  │ PERG engineered + clinical (age/sex/VA/logMAR) │
(one vector) ────▶└────────────────────────────────────────────────┘
        │                 │              │             │            │
        ▼                 ▼              ▼             ▼            ▼
   ┌────────┐        ┌────────┐     ┌────────┐    ┌────────┐   ┌────────┐
   │  LR    │        │  SVM   │     │  KNN   │    │  RF    │   │  GBM   │   (Level‑0)
   └────────┘        └────────┘     └────────┘    └────────┘   └────────┘
        │                 │              │             │            │
        └───────┬─────────┴───────┬──────┴───────┬─────┴───────┬────┘
                ▼                 ▼              ▼            ▼
        p_LR (prob)        p_SVM (prob)    p_KNN (prob)  p_RF (prob)  p_GBM (prob)

                         ┌──────────────────────────────────────────┐
                         │ Meta-learner: Logistic Regression        │(Level‑1)
                         │ input = [p_LR, p_SVM, p_KNN, p_RF, p_GBM]│
                         └──────────────────────────────────────────┘
                                         │
                                         ▼
                               Final P(pathological)
```
    
    Key point: the models are combined after they each predict; Level‑1 learns how to weight their probabilities.

## 3) Model 3 Two-layer cascade classifie

```text
Eye-level sample
(PERG engineered features + clinical features)
          │
          ▼
 ┌─────────────────────────────────────┐
 │ Stage 1: Fast gate model            │
 │ (e.g., XGBoost / quick classifier)  │
 └─────────────────────────────────────┘
          │
          │ outputs probability p1 = P(pathological)
          ▼
   ┌────────────────────────────────────────────┐
   │ Decision rule (confidence / threshold):    │
   │                                            │
   │ If p1 is very low  → classify Normal       │
   │ If p1 is very high → classify Pathological │
   │ Else (uncertain)   → send to Stage 2       │
   └────────────────────────────────────────────┘
          │                     │
          │(easy/clear cases)   │(borderline cases)
          ▼                     ▼
  Final decision         ┌───────────────────────────────┐
                         │ Stage 2: Stronger model       │
                         │ (Model 1 dual-stream NN       │
                         │  OR Model 2 stacking ensemble)│
                         └───────────────────────────────┘
                                      │
                                      ▼
                               Final decision
```    

    - Stage 1 is a quick screener to handle obvious cases cheaply/fast.
    - Stage 2 is the more powerful model used only when Stage 1 is unsure.