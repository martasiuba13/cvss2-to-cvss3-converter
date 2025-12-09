import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# Wczytanie pliku z danymi


file_path = r"C:\Users\Marta\Desktop\Input_data_cvss\data_word_50.csv"
df = pd.read_csv(file_path, sep=r"\s+", header=None)

print("Kształt danych (wiersze, kolumny):", df.shape)

# -----------------------------------------------
# Podział danych na X (cechy) i y_UI

# X: 57 cech (0–56)
X = df.iloc[:, 0:57]

# UI: kolumna 60 (CVSS 3.1) – 2 klasy: 0 (NONE), 1 (REQUIRED)
y = df.iloc[:, 60]

print("\nKształt X:", X.shape)
print("Kształt y (UI):", y.shape)

print("\nLiczność klas w UI (cały zbiór):")
print(y.value_counts())

#--------------------------------
#Podział na zbiór treningowy i testowy

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=3179,   # 73179 - 70000
    shuffle=True,
    random_state=42
)

print("\nRozmiary powstałych zbiorów:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nLiczność klas w y_train (UI):")
print(y_train.value_counts())

# ---------------------
# Skalowanie danych

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPrzykładowy wiersz przed skalowaniem:")
print(X_train.iloc[0])

print("\nPrzykładowy wiersz po skalowaniu:")
print(X_train_scaled[0])

# ------------------------------------------------
# Budowa modelu sieci neuronowej dla UI (2 klasy)


model = Sequential([
    Dense(64, activation='relu', input_shape=(57,)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')   # 2 klasy: 0,1
])

print("\nPodsumowanie modelu (UI):")
model.summary()

# ------------------
# Kompilacja

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # etykiety 0/1
    metrics=['accuracy']
)

# ----------------------
# EarlyStopping


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# Wagi klas (UI) – jeśli dane niezbalansowane

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weight_dict = dict(zip(classes, class_weights))

print("\nWagi klas (UI):")
print(class_weight_dict)

# --------------------------
# Trenowanie modelu (UI)

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# --------------------------
# Ocena na zbiorze testowym

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print("\nWyniki na zbiorze testowym dla UI:")
print("Loss (błąd):", test_loss)
print("Accuracy (dokładność):", test_accuracy)

# -------------------------
# Macierz pomyłek i raport


y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nMacierz pomyłek (UI):")
print(cm)

print("\nRaport klasyfikacji (UI):")
print(classification_report(y_test, y_pred))

# ----------------------
# Zapis modelu UI do pliku .h5

model.save("model_ui.h5")
print("\nModel UI zapisano do pliku: model_ui.h5")
