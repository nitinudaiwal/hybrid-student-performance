"""
Hybrid Intelligent System – Student Performance Predictor
==========================================================
Author  : Nitin Udaiwal
Course  : Introduction to Intelligent Systems

Architecture:
  Step 1 → Fuzzy Logic converts raw inputs into crisp fuzzy scores
           (Attendance → Attendance Score, Assignments → Assignment Score,
            Test Marks → Test Score)
  Step 2 → Neural Network learns from the fuzzy scores to classify
           Performance Level: Poor / Average / Good

Inputs (raw):
  - Attendance   : 0–100 %
  - Assignment   : 0–10 marks
  - Test Marks   : 0–100 marks

Output:
  - Performance Level : 0=Poor, 1=Average, 2=Good
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# ═══════════════════════════════════════════════════
# STEP 1 — FUZZY LOGIC MODULE
# ═══════════════════════════════════════════════════

def fuzzy_attendance_score(attendance):
    """Convert attendance % → fuzzy performance score (0-1)"""
    low    = fuzz.trimf(np.arange(0, 101, 1), [0, 0, 60])
    medium = fuzz.trimf(np.arange(0, 101, 1), [50, 70, 85])
    high   = fuzz.trimf(np.arange(0, 101, 1), [75, 100, 100])
    score = (
        fuzz.interp_membership(np.arange(0, 101, 1), low,    attendance) * 0.2 +
        fuzz.interp_membership(np.arange(0, 101, 1), medium, attendance) * 0.5 +
        fuzz.interp_membership(np.arange(0, 101, 1), high,   attendance) * 1.0
    )
    # normalize
    total = (
        fuzz.interp_membership(np.arange(0, 101, 1), low,    attendance) +
        fuzz.interp_membership(np.arange(0, 101, 1), medium, attendance) +
        fuzz.interp_membership(np.arange(0, 101, 1), high,   attendance)
    )
    return score / total if total > 0 else 0.0


def fuzzy_assignment_score(marks):
    """Convert assignment marks (0-10) → fuzzy score (0-1)"""
    low    = fuzz.trimf(np.arange(0, 11, 1), [0, 0, 5])
    medium = fuzz.trimf(np.arange(0, 11, 1), [3, 6, 8])
    high   = fuzz.trimf(np.arange(0, 11, 1), [6, 10, 10])
    score = (
        fuzz.interp_membership(np.arange(0, 11, 1), low,    marks) * 0.2 +
        fuzz.interp_membership(np.arange(0, 11, 1), medium, marks) * 0.5 +
        fuzz.interp_membership(np.arange(0, 11, 1), high,   marks) * 1.0
    )
    total = (
        fuzz.interp_membership(np.arange(0, 11, 1), low,    marks) +
        fuzz.interp_membership(np.arange(0, 11, 1), medium, marks) +
        fuzz.interp_membership(np.arange(0, 11, 1), high,   marks)
    )
    return score / total if total > 0 else 0.0


def fuzzy_test_score(marks):
    """Convert test marks (0-100) → fuzzy score (0-1)"""
    low    = fuzz.trimf(np.arange(0, 101, 1), [0, 0, 50])
    medium = fuzz.trimf(np.arange(0, 101, 1), [35, 60, 80])
    high   = fuzz.trimf(np.arange(0, 101, 1), [65, 100, 100])
    score = (
        fuzz.interp_membership(np.arange(0, 101, 1), low,    marks) * 0.2 +
        fuzz.interp_membership(np.arange(0, 101, 1), medium, marks) * 0.5 +
        fuzz.interp_membership(np.arange(0, 101, 1), high,   marks) * 1.0
    )
    total = (
        fuzz.interp_membership(np.arange(0, 101, 1), low,    marks) +
        fuzz.interp_membership(np.arange(0, 101, 1), medium, marks) +
        fuzz.interp_membership(np.arange(0, 101, 1), high,   marks)
    )
    return score / total if total > 0 else 0.0


def apply_fuzzy_module(attendance, assignment, test):
    """Apply all three fuzzy functions and return fuzzy feature vector."""
    fa = fuzzy_attendance_score(attendance)
    fb = fuzzy_assignment_score(assignment)
    ft = fuzzy_test_score(test)
    return np.array([fa, fb, ft])


# ═══════════════════════════════════════════════════
# STEP 2 — GENERATE SYNTHETIC DATASET
# ═══════════════════════════════════════════════════
def label_from_raw(att, asn, test):
    """Rule-based labeling for synthetic data generation."""
    score = (att / 100) * 0.3 + (asn / 10) * 0.3 + (test / 100) * 0.4
    if score < 0.40:
        return 0   # Poor
    elif score < 0.70:
        return 1   # Average
    else:
        return 2   # Good

N = 800
attendance_data  = np.random.uniform(30, 100, N)
assignment_data  = np.random.uniform(0,  10,  N)
test_data        = np.random.uniform(20, 100, N)

# Fuzzy transform all samples
X_fuzzy = np.array([
    apply_fuzzy_module(a, b, t)
    for a, b, t in zip(attendance_data, assignment_data, test_data)
])
y = np.array([
    label_from_raw(a, b, t)
    for a, b, t in zip(attendance_data, assignment_data, test_data)
])

print(f"Dataset shape : {X_fuzzy.shape}")
print(f"Class counts  : Poor={np.sum(y==0)}, Average={np.sum(y==1)}, Good={np.sum(y==2)}")

# ═══════════════════════════════════════════════════
# STEP 3 — NEURAL NETWORK
# ═══════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X_fuzzy, y, test_size=0.2, random_state=42, stratify=y
)

model = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(8,  activation='relu'),
    keras.layers.Dense(3,  activation='softmax')   # 3 output classes
], name="StudentPerformanceNN")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# ═══════════════════════════════════════════════════
# STEP 4 — EVALUATE
# ═══════════════════════════════════════════════════
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy : {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
labels = ['Poor', 'Average', 'Good']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# ═══════════════════════════════════════════════════
# STEP 5 — PLOTS
# ═══════════════════════════════════════════════════

# --- (a) Training curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Neural Network Training – Hybrid System', fontsize=13, fontweight='bold')

axes[0].plot(history.history['accuracy'],     label='Train Acc', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Acc',   linewidth=2, linestyle='--')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train Loss', linewidth=2, color='red')
axes[1].plot(history.history['val_loss'], label='Val Loss',   linewidth=2, linestyle='--', color='orange')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# --- (b) Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix – Student Performance\n(Hybrid Fuzzy + Neural Network)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# --- (c) Fuzzy membership function plots ---
x_att = np.arange(0, 101, 1)
x_asn = np.arange(0, 11, 1)
x_tst = np.arange(0, 101, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Fuzzy Membership Functions – Student Performance System', fontsize=13, fontweight='bold')

# Attendance
axes[0].plot(x_att, fuzz.trimf(x_att, [0, 0, 60]),   'b-',  label='Low',    lw=2)
axes[0].plot(x_att, fuzz.trimf(x_att, [50, 70, 85]),  'g--', label='Medium', lw=2)
axes[0].plot(x_att, fuzz.trimf(x_att, [75, 100, 100]),'r-',  label='High',   lw=2)
axes[0].set_title('Attendance (%)'); axes[0].set_xlabel('Attendance %')
axes[0].set_ylabel('Membership'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Assignment
axes[1].plot(x_asn, fuzz.trimf(x_asn, [0, 0, 5]),   'b-',  label='Low',    lw=2)
axes[1].plot(x_asn, fuzz.trimf(x_asn, [3, 6, 8]),   'g--', label='Medium', lw=2)
axes[1].plot(x_asn, fuzz.trimf(x_asn, [6, 10, 10]), 'r-',  label='High',   lw=2)
axes[1].set_title('Assignment Marks (0–10)'); axes[1].set_xlabel('Marks')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Test Marks
axes[2].plot(x_tst, fuzz.trimf(x_tst, [0, 0, 50]),    'b-',  label='Low',    lw=2)
axes[2].plot(x_tst, fuzz.trimf(x_tst, [35, 60, 80]),   'g--', label='Medium', lw=2)
axes[2].plot(x_tst, fuzz.trimf(x_tst, [65, 100, 100]), 'r-',  label='High',   lw=2)
axes[2].set_title('Test Marks (0–100)'); axes[2].set_xlabel('Marks')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fuzzy_membership_student.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ All plots saved!")

# ═══════════════════════════════════════════════════
# STEP 6 — DEMO PREDICTION
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("  DEMO PREDICTIONS")
print("="*60)
demo_cases = [
    (40, 3, 35, "Poor student"),
    (70, 6, 62, "Average student"),
    (95, 9, 88, "Good student"),
    (55, 5, 50, "Borderline case"),
]
for att, asn, test, desc in demo_cases:
    fv = apply_fuzzy_module(att, asn, test).reshape(1, -1)
    pred = np.argmax(model.predict(fv, verbose=0))
    conf = np.max(model.predict(fv, verbose=0)) * 100
    print(f"  {desc:<20} → {labels[pred]:<8}  (confidence: {conf:.1f}%)")
    print(f"    Input: Att={att}%, Asn={asn}/10, Test={test}/100")
    print(f"    Fuzzy vector: {apply_fuzzy_module(att, asn, test).round(2)}")
print("="*60)
