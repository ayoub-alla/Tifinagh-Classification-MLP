import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

from src.model import MultiClass_NN

import joblib


X_train = np.load('data/ready/X_train.npy') 
X_val = np.load('data/ready/X_val.npy')
X_test = np.load('data/ready/X_test.npy')
y_train = np.load('data/ready/y_train.npy')
y_val = np.load('data/ready/y_val.npy')
y_test = np.load('data/ready/y_test.npy')


# Encoder les etiquettes en one-hot pour la classification multiclasse
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = np.array(one_hot_encoder.fit_transform(y_train.reshape (-1, 1)))
y_val_one_hot = np.array(one_hot_encoder.transform(y_val.reshape(-1, 1)))
y_test_one_hot = np.array(one_hot_encoder.transform(y_test.reshape(-1, 1)))

# Verifier que les tableaux one-hot sont des NumPy arrays
assert isinstance(y_train_one_hot, np.ndarray), "y_train_one_hot must be a numpy array"
assert isinstance (y_val_one_hot, np.ndarray), "y_val_one_hot must be a numpy array"
assert isinstance (y_test_one_hot, np.ndarray), "y_test_one_hot must be a numpy array"

# Creer et entrainer le modele
layer_sizes = [X_train.shape [1], 64, 32, 33] # 64 et 32 neurones caches, 33 classes
#L2 used 0.15 
nn = MultiClass_NN(layer_sizes, learning_rate=0.08 , lambda_reg=0.15 )
train_losses, val_losses, train_accuracies, val_accuracies = nn.train (
    X_train, y_train_one_hot, X_val, y_val_one_hot, epochs=100, batch_size=32
)


# Predictions et evaluation
label_encoder = joblib.load('artifacts/label_encoder.joblib')

y_pred = nn.predict(X_test)
print("\nRapport de classification (Test set) :")
print (classification_report (y_test, y_pred, target_names=label_encoder.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap (cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion (Test set)')
plt.xlabel('Predit')
plt.ylabel('Reel')
plt.savefig('assets/confusion_matrix.png')
plt.close()

# Courbes de perte et d'accuracy
fig, (ax1, ax2) = plt.subplots (1, 2, figsize=(12, 5))

# Courbe de perte
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Validation Loss')
ax1.set_title('Courbe de perte')
ax1.set_xlabel('Epoque')
ax1.set_ylabel('Perte')
ax1.legend ()

# Courbe d'accuracy
ax2.plot(train_accuracies, label='Train Accuracy')
ax2.plot(val_accuracies, label='Validation Accuracy')
ax2.set_title('Courbe de precision')
ax2.set_xlabel('Epoque')
ax2.set_ylabel('Precision')
ax2.legend()

plt.tight_layout()
fig.savefig('assets/loss_accuracy_plot.png')
plt.close()

X_full = np.concatenate([X_train, X_val], axis=0)
y_full = np.concatenate([y_train, y_val], axis=0)

kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_val_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
    X_tr, X_v = X_full[train_idx], X_full[val_idx]
    y_tr, y_v = y_full[train_idx], y_full[val_idx]

    y_tr_oh = one_hot_encoder.transform(y_tr.reshape(-1, 1))
    y_v_oh = one_hot_encoder.transform(y_v.reshape(-1, 1))

    nn = MultiClass_NN(layer_sizes, learning_rate=0.08, lambda_reg=0.0)
    nn.train(X_tr, y_tr_oh, X_v, y_v_oh, epochs=100, batch_size=32)

    val_pred = nn.forward(X_v)
    acc = nn.compute_accuracy(y_v_oh, val_pred)
    fold_val_accuracies.append(acc)
    print(f"Fold {fold+1}: Val Acc = {acc:.4f}")

print(f"\nMean CV Accuracy: {np.mean(fold_val_accuracies):.4f} ± {np.std(fold_val_accuracies):.4f}")