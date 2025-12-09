import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE   #

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Wczytanie danych

file_path = r"C:\Users\Marta\Desktop\Input_data_cvss\data_word_50.csv"
df = pd.read_csv(file_path, sep=r"\s+", header=None)

print("Kształt danych (wiersze, kolumny):", df.shape)

# ----------------------------------
# 2. X i y_A (Availability, kolumna 64)

# X: 57 cech (0–56)
X = df.iloc[:, 0:57]

# A: kolumna 64 (CVSS 3.1) – 3 klasy: 0 (NONE), 1 (LOW), 2 (HIGH)
y = df.iloc[:, 64]

print("\nKształt X:", X.shape)
print("Kształt y (A):", y.shape)

print("\nLiczność klas w A (cały zbiór):")
print(y.value_counts())

# ---------------------------
#  trening i test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=3179,    # 73179 - 70000
    shuffle=True,
    random_state=42
)

print("\nRozmiary zbiorów:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nLiczność klas w y_train (A) PRZED SMOTE:")
print(y_train.value_counts())

# -------------------------------
# Skalowanie danych (ylko X)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------
#  SMOTE

# Tworzymy obiekt SMOTE
smote = SMOTE(random_state=42)

# Uwaga: SMOTE stosujemy TYLKO na zbiorze treningowym
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("\nLiczność klas w y_train_po_SMOTE (A):")
print(pd.Series(y_train_bal).value_counts())

#--------------------------------------------------
#  Budowa modelu sieci neuronowej dla A (3 klasy)
model = Sequential([
    Dense(64, activation='relu', input_shape=(57,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')   # 3 klasy: 0,1,2
])

print("\nPodsumowanie modelu (A, SMOTE):")
model.summary()

# ------------------
# Kompilacja modelu

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------------------------
# EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ---------------------------------------
# Trenowanie modelu na danych po SMOTE

history = model.fit(
    X_train_bal,
    y_train_bal,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


# Ocena na zbiorze testowym (nie zmienionym przez SMOTE)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print("\nWyniki na zbiorze testowym dla A (po SMOTE):")
print("Loss (błąd):", test_loss)
print("Accuracy (dokładność):", test_accuracy)

# ----------------------------------------
# Macierz pomyłek i raport klasyfikacji

y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nMacierz pomyłek (A, po SMOTE):")
print(cm)

print("\nRaport klasyfikacji (A, po SMOTE):")
print(classification_report(y_test, y_pred))



model.save("model_a.h5")
print("\nModel A zapisano do pliku: model_a.h5")
