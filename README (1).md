# 🎓 Hybrid Intelligent System – Student Performance Predictor

**Course**: Introduction to Intelligent Systems  
**Assignment**: Q2  
**Author**: Nitin Udaiwal  
**Reg. No**: 23FE10CCE00056  
**University**: Manipal University Jaipur  

---

## 📌 Problem Statement

Develop a **hybrid intelligent system** combining fuzzy logic and neural networks to predict student performance level.

**Inputs:**
- Attendance (0–100 %)
- Assignment Marks (0–10)
- Test Marks (0–100)

**Output:**
- Performance Level: `Poor` / `Average` / `Good`

---

## 🏗️ System Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   RAW INPUTS    │────▶│   FUZZY LOGIC MODULE  │────▶│  NEURAL NETWORK │
│                 │     │                       │     │                 │
│ • Attendance %  │     │ Membership Functions: │     │ Input  Layer: 3 │
│ • Assignment Mk │     │ • Low / Medium / High │     │ Hidden Layer: 16│
│ • Test Marks    │     │                       │     │ Hidden Layer: 8 │
└─────────────────┘     │ → Fuzzy Score (0–1)   │     │ Output Layer: 3 │
                        └──────────────────────┘     └────────┬────────┘
                                                              │
                                                    ┌─────────▼──────────┐
                                                    │  PERFORMANCE LEVEL  │
                                                    │   Poor/Average/Good │
                                                    └────────────────────┘
```

---

## 🔵 Step 1 – Fuzzy Logic Module

### Purpose
Transform raw numerical inputs into normalized **fuzzy scores** (0.0 to 1.0) that represent the quality/strength of each parameter.

### Fuzzy Sets for Each Input

**Attendance (0–100%):**
| Linguistic | MF Type | Parameters |
|------------|---------|------------|
| Low        | trimf   | [0, 0, 60] |
| Medium     | trimf   | [50, 70, 85] |
| High       | trimf   | [75, 100, 100] |

**Assignment Marks (0–10):**
| Linguistic | MF Type | Parameters |
|------------|---------|------------|
| Low        | trimf   | [0, 0, 5] |
| Medium     | trimf   | [3, 6, 8] |
| High       | trimf   | [6, 10, 10] |

**Test Marks (0–100):**
| Linguistic | MF Type | Parameters |
|------------|---------|------------|
| Low        | trimf   | [0, 0, 50] |
| Medium     | trimf   | [35, 60, 80] |
| High       | trimf   | [65, 100, 100] |

### Fuzzy Rules (Applied per dimension)

| If Linguistic Label is... | Then Fuzzy Score Weight |
|---------------------------|------------------------|
| Low                       | 0.2 (weak)             |
| Medium                    | 0.5 (moderate)         |
| High                      | 1.0 (strong)           |

**Output:** Weighted average = normalized fuzzy score per input dimension

---

## 🟠 Step 2 – Neural Network Module

### Architecture

```
Input Layer  (3 neurons)  → Fuzzy scores [att_score, asn_score, test_score]
Hidden Layer (16 neurons) → ReLU activation
Dropout      (20%)        → Regularization (prevents overfitting)
Hidden Layer (8 neurons)  → ReLU activation
Output Layer (3 neurons)  → Softmax → [P(Poor), P(Average), P(Good)]
```

### Training Details

| Parameter    | Value                         |
|-------------|-------------------------------|
| Optimizer   | Adam                          |
| Loss        | Sparse Categorical Crossentropy |
| Epochs      | 60                            |
| Batch Size  | 32                            |
| Train/Test  | 80% / 20%                     |
| Dataset     | 800 synthetic samples         |
| Accuracy    | ~80%+                         |

---

## 🔗 Integration: How Fuzzy + Neural Network Work Together

```
Fuzzy Logic acts as a "smart preprocessor":
  ✅ Handles uncertainty in raw input data
  ✅ Converts linguistic variables to numeric scores
  ✅ Reduces noise and standardizes ranges

Neural Network acts as the "classifier":
  ✅ Learns complex patterns from fuzzy-processed data
  ✅ Generalizes to unseen student profiles
  ✅ Outputs probability distribution over classes
```

**Why hybrid?**  
- Fuzzy alone: Good at handling uncertainty, but limited learning  
- Neural network alone: Struggles with ambiguous/overlapping data  
- **Hybrid**: Best of both — interpretable fuzzy front-end + powerful learning back-end

---

## 📊 Sample Predictions

| Student Profile          | Att% | Asn | Test | Prediction | Confidence |
|--------------------------|------|-----|------|------------|------------|
| Poor student             | 40   | 3   | 35   | Average    | ~72%       |
| Average student          | 70   | 6   | 62   | Average    | ~75%       |
| Good student             | 95   | 9   | 88   | Good       | ~82%       |
| Borderline case          | 55   | 5   | 50   | Average    | ~75%       |

---

## 🛠️ How to Run

```bash
pip install scikit-fuzzy tensorflow scikit-learn numpy matplotlib
python student_performance_hybrid.py
```

**Outputs generated:**
- `fuzzy_membership_student.png` – Membership function plots
- `training_curves.png`         – Accuracy & Loss curves
- `confusion_matrix.png`        – Classification results

---

## 📁 Files

```
student_performance_hybrid/
├── student_performance_hybrid.py    # Main Python implementation
├── fuzzy_membership_student.png     # Fuzzy MF visualizations
├── training_curves.png              # NN training history
├── confusion_matrix.png             # Evaluation results
└── README.md                        # This documentation
```
