import os
import pandas as pd
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


# Definir le chemin vers le dossier decompresse
data_dir = os.path.join(os.getcwd(), 'data/amhcd-data-64/tifinagh-images/')
print(data_dir)
project_dir = os.path.join(os.getcwd(), '/')
print(project_dir)

# crier le fichier CSV contenant les etiquettes
# Alternative construire un DataFrame a partir des dossiers
image_paths = []
labels = []
for label_dir in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label_dir)
    if os.path.isdir (label_path):
        for img_name in os.listdir (label_path):
            image_paths.append(os.path.join(label_path, img_name))
            labels.append(label_dir)
labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})


# Verifier le DataFrame
assert not labels_df.empty, "No data loaded. Check dataset files."
print (f"Loaded {len(labels_df)} samples with {labels_df['label'].nunique()} unique classes.")

# Encoder les etiquettes
label_encoder = LabelEncoder()
labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
num_classes = len(label_encoder.classes_)
print("num_classes :" , num_classes)

joblib.dump(label_encoder,  'artifacts/label_encoder.joblib')

# Fonction pour charger et pretraiter une image
def load_and_preprocess_image(image_path, target_size=(32,32)):

    """
    Load and preprocess an image convert to grayscale, resize, normalize
    """
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    assert img is not None, f"Failed to load image: {image_path}"
    img = cv2.resize(img, target_size)
    img = img.astype (np.float32) / 255.0 # Normalisation

    return img.flatten() # Aplatir pour le reseau de neurones

# Charger toutes les images
X = np.array([load_and_preprocess_image(os.path.join(data_dir, path)) for path in labels_df ['image_path']])
y = labels_df['label_encoded'].values

# Verifier les dimensions
assert X.shape [0] == y.shape [0], "Mismatch between number of images and labels"
assert X.shape [1] == 32 * 32, f"Expected flattened image size of {32*32}, got {X.shape[1]}"

# Diviser en ensembles d'entrainement, validation et test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# Convertir explicitement en NumPy arrays
X_train = np.save(project_dir+'data/ready/X_train.npy', np.array(X_train)) 
X_val = np.save(project_dir+'data/ready/X_val.npy', np.array(X_val))
X_test = np.save(project_dir+'data/ready/X_test.npy', np.array(X_test))
y_train = np.save(project_dir+'data/ready/y_train.npy',np.array(y_train))
y_val = np.save(project_dir+'data/ready/y_val.npy', np.array(y_val))
y_test = np.save(project_dir+'data/ready/y_test.npy',np.array(y_test))


