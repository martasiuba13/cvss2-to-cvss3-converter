import pandas as pd  # biblioteka do pracy z danymi w formie tabel
import numpy as np   #biblioteka do pracy z tablicami liczbowymi (macierze)

# stadard w uczeniu maszynowym
from sklearn.model_selection import train_test_split  #funkcja, która dzieli dane na część do uczenia i część do testowania.
from sklearn.preprocessing import StandardScaler  #narzędzie do skalowania cech
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# Wczytanie pliku z danymi

file_path = r"C:\Users\Marta\Desktop\Input_data_cvss\data_word_50.csv"
df = pd.read_csv(file_path, sep=r"\s+", header=None)

print("Kształt danych (wiersze, kolumny):", df.shape)

# ------------------------------
# Podział danych na X (cechy) i y_S (Scope v3.1)

# X: 57 cech (kolumny 0–56)
X = df.iloc[:, 0:57]

# S: kolumna 61 (CVSS 3.1) – 2 klasy: 0 (UNCHANGED), 1 (CHANGED)
y = df.iloc[:, 61]

print("\nKształt X:", X.shape)
print("Kształt y (S):", y.shape)

print("\nLiczność klas w S (cały zbiór):")
print(y.value_counts())

#-----------------------------------------
# Podział na zbiór treningowy i testowy

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

print("\nLiczność klas w y_train (S):")
print(y_train.value_counts())

# -----------------------
# Skalowanie danych

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPrzykładowy wiersz przed skalowaniem:")
print(X_train.iloc[0])

print("\nPrzykładowy wiersz po skalowaniu:")
print(X_train_scaled[0])

#--------------------------------------------------------
# 5. Budowa modelu sieci neuronowej dla S (2 klasy)

model = Sequential([
    Dense(64, activation='relu', input_shape=(57,)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')   # 2 klasy: 0,1
])

print("\nPodsumowanie modelu (S):")
model.summary()

#---------------------
# Kompilacja

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------
#  EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

#-----------------
# Wagi klas (S)

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weight_dict = dict(zip(classes, class_weights))

print("\nWagi klas (S):")
print(class_weight_dict)

# --------------------------
# Trenowanie modelu (S)

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

# -------------------------------
# Ocena na zbiorze testowym

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print("\nWyniki na zbiorze testowym dla S:")
print("Loss (błąd):", test_loss)
print("Accuracy (dokładność):", test_accuracy)

# ---------------------------
# Macierz pomyłek i raport

y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nMacierz pomyłek (S):")
print(cm)

print("\nRaport klasyfikacji (S):")
print(classification_report(y_test, y_pred))

#-------------------------------
# Zapis modelu S do pliku .h5

model.save("model_s.h5")
print("\nModel S zapisano do pliku: model_s.h5")
