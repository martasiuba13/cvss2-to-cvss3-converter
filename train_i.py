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

# Wczytanie danych
file_path = r"C:\Users\Marta\Desktop\Input_data_cvss\data_word_50.csv"
df = pd.read_csv(file_path, sep=r"\s+", header=None)
print("Kształt danych (wiersze, kolumny):", df.shape)

# 2. X i y_I (Integrity)
X = df.iloc[:, 0:57]
y = df.iloc[:, 63]   # kolumna 63 = I

print("\nKształt X:", X.shape)
print("Kształt y (I):", y.shape)
print("\nLiczność klas w I (cały zbiór):")
print(y.value_counts())

# Train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=3179,
    shuffle=True,
    random_state=42
)

print("\nRozmiary zbiorów:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print("\nLiczność klas w y_train (I):")
print(y_train.value_counts())

# Skalowanie
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model (3 klasy)
model = Sequential([
    Dense(64, activation='relu', input_shape=(57,)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

print("\nPodsumowanie modelu (I):")
model.summary()

# Kompilacja
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Wagi klas
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))
print("\nWagi klas (I):")
print(class_weight_dict)

# Trening
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

# Test
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print("\nWyniki na zbiorze testowym dla I:")
print("Loss (błąd):", test_loss)
print("Accuracy (dokładność):", test_accuracy)

#  Macierz + raport
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nMacierz pomyłek (I):")
print(cm)

print("\nRaport klasyfikacji (I):")
print(classification_report(y_test, y_pred))

#----------------------------------
# Zapis modelu I do pliku .h5


model.save("model_i.h5")
print("\nModel I zapisano do pliku: model_i.h5")
