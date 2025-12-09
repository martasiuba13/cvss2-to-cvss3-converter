import joblib  # DO ZAPISU / ODCZYTU SKALERA (tylko raz potrzebny)


import pandas as pd  # biblioteka do pracy z danymi w formie tabel
import numpy as np   #biblioteka do pracy z tablicami liczbowymi (macierze)

# stadard w uczeniu maszynowym
from sklearn.model_selection import train_test_split  #funkcja, która dzieli dane na część do uczenia i część do testowania.
from sklearn.preprocessing import StandardScaler  #narzędzie do skalowania cech
from sklearn.metrics import confusion_matrix, classification_report #funkcja do liczenia macierzy pomyłek/ wypisuje precision, recall, f1-score dla każdej klasy

from imblearn.over_sampling import SMOTE   # oversampling klas rzadkich rozwiązuje problem niezbalansowanych danych

# Sieci neuronowe

import tensorflow as tf
from tensorflow.keras.models import Sequential  # model sekwencyjny
from tensorflow.keras.layers import Dense  # warstwa neuronów, gdzie każdy neuron ma połączenie ze wszystkimi wejściami
from tensorflow.keras.callbacks import EarlyStopping # mechanizm, który przerywa uczenie, gdy model przestaje się poprawiać na zbiorze walidacyjnym – żeby nie przeuczyć modelu

# -------------------------
# Wczytanie danych


file_path = r"C:\Users\Marta\Desktop\Input_data_cvss\data_word_50.csv"
df = pd.read_csv(file_path, sep=r"\s+", header=None)

#sep=r"\s+" - separator to dowolna ilość spacji
#plik nie na nagłówków z nazwami oklumn wiec traktujemy wszystko jako dane

print("Kształt danych (wiersze, kolumny):", df.shape)

#df to tera tabela o rozmiarze [73179,66]


#---------------------------------------
# X i y_AV (Attack Vector, kolumna 57)

# X: 57 cech (0–56)
X = df.iloc[:, 0:57]

# AV: kolumna 57 – 4 klasy (0,1,2,3)
y = df.iloc[:, 57]

print("\nKształt X:", X.shape)  #pokazuje ile jest wierszy i kolumn w zbiorze cech
print("Kształt y (AV):", y.shape)  # pokazuje ile etykiet

print("\nLiczność klas w AV (cały zbiór):")
print(y.value_counts()) # ile jest próbek w każdej klasie AV

# ----------------------------------------
# Podział na zbiór treningowy i testowy


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=3179,    # 73179 - 70000 zapisana na sztywno liczba próbek testowych
    shuffle=True,      # train_test_split – automatycznie miesza dane (shuffle=True) i dzieli je na część do uczenia i testowania.
    random_state=42    # zebyby zapewnić pełną powtarzalność eksperymentów
)


print("\nRozmiary zbiorów:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nLiczność klas w y_train (AV) PRZED SMOTE:")
print(y_train.value_counts())  #  liczność klas przed zastosowaniem SMOTE (pokazuje dowody ze kalsy są nierówne)

# ----------------------------------------------
# Skalowanie danych

scaler = StandardScaler() # utworzenie obiektu skalującego
X_train_scaled = scaler.fit_transform(X_train) # fit oblicza średnią i odchylenie dla każdej kolumny na zbiorze treningowym , transform – przekształca liczby tak, żeby: średnia ≈ 0,odchylenie ≈ 1.
X_test_scaled = scaler.transform(X_test) # do zbioru testowego stosujesz te same wyliczone parametry zeby nie uczyć skalera drugi raz

# ---------------------------------------------------
# SMOTE – balansowanie klas na zbiorze treningowym

smote = SMOTE(random_state=42) #konfiguracja generatora SMOTE
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train) # na podsatwie danych treningowych i ich klas (y_train) smote tworzy nowe, sztuczne przykłądy mniejszyk klas, aby każda miała podobna liczebnosc
# daltego mamy X_train_bal – zbalansowane cechy treningowe,y_train_bal – zbalansowane etykiety

print("\nLiczność klas w y_train_po_SMOTE (AV):")
print(pd.Series(y_train_bal).value_counts())

# -------------------------------------------------
# Budowa modelu sieci neuronowej dla AV (4 klasy)

# mówi ze ta siec jest po prostu listą wartsw
model = Sequential([
    Dense(64, activation='relu', input_shape=(57,)),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')   # 4 klasy: 0,1,2,3
])
# Na wejsciu 57 cech (tyle ile kolumn)
#64 neurony
# nieliniowa funkcja aktywacji, dzieki której siec moze uczyć się skomplikowanych zależnści
# Dense(32, activation='relu') - druga wartwa ukrywa, 32 nuerony
#Dense(4, activation='softmax') - wartwa wyjsciowa, 4 neurony bo mam 4 klasy, zmienia wyjscie n 4 prawdopodobienstwa


print("\nPodsumowanie modelu (AV, SMOTE):")
model.summary()
#model.summary() wypisuje tabelkę z liczbą parametrów
# -------------------------------------------------------
# Kompilacja modelu


model.compile(
    optimizer='adam', #algorytm optymalizacji (szuka najlepszych wag)
    loss='sparse_categorical_crossentropy',  # etykiety 0..3 funkcja kosztu dla problemu klasyfikacji wieloklasowej, gdy etykiety są liczbami calkowitymi
    metrics=['accuracy'] # sledzi dokladnosc w trakcie uczenia i testowania
)

# -------------------------------------------------------------------
# EarlyStopping

#monitor='val_loss' - obserwacja błądów na zbiorze walidacyjnym
#patience=3 – jeśli przez 3 kolejne epoki błąd się nie poprawia, uczenie zostanie przerwane.
#restore_best_weights=True – po zakończeniu uczenia model wraca do wag z najlepszej epoki.

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# --------------------------------------------------
# Trenowanie modelu (AV) na danych po SMOTE


history = model.fit(
    X_train_bal,
    y_train_bal, # uczy sie na zbalansowanych danych po SMOTE
    epochs=20,   #maksymalnie 20 przebiegów po wszystkich danych
    batch_size=128, #na raz model przetwarza 128 przykładów zanim zaktualizuje wagi
    validation_split=0.2, #20% danych treningowych odkładanych jest jako zbiór walidacyjny – do monitorowania val_loss
    callbacks=[early_stop], # uzycie tego wczesniejszego zatrzymania
    verbose=1 # postęp uczenia jest wypisywany w konsoli.
)

# ---------------------------------------------------------
# Ocena na zbiorze testowym (niezbalansowanym)


test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
#.evaluate(...) – podaje modelowi dane, których nie widział podczas uczenia (X_test_scaled, y_test)

print("\nWyniki na zbiorze testowym dla AV (po SMOTE):")
print("Loss (błąd):", test_loss)  #błąd
print("Accuracy (dokładność):", test_accuracy)  #procent poprawnych klasyfikacji

# ------------------------------------------
# Macierz pomyłek i raport


y_pred_proba = model.predict(X_test_scaled)  #  dla każdego przykładu z testu model zwraca 4 prawdopodobieństwa (dla klas 0,1,2,3)
y_pred = np.argmax(y_pred_proba, axis=1)  #   wybierasz klasę z największym prawdopodobieństwem

cm = confusion_matrix(y_test, y_pred) # buduje macierz pomyłek
print("\nMacierz pomyłek (AV, po SMOTE):")
print(cm)

print("\nRaport klasyfikacji (AV, po SMOTE):")
print(classification_report(y_test, y_pred))

#classification_report wypisuje:
#precision – z tych, które model oznaczył jako daną klasę, jaki procent był faktycznie poprawny
#recall - z tych, które powinny być daną klasą, jaki procent model poprawnie wykrył
#f1-score – średnia harmoniczna z precision i recall (miara jakości klasy)
#support – ile przykładów danej klasy w testowych dan


# Zapis skalera do pliku (robimy to tylko raz, np. w AV)

joblib.dump(scaler, "scaler.pkl")
print("\nSkaler zapisano do pliku: scaler.pkl")

# Zapis modelu AV
model.save("model_av.h5")  # albo model_av.h5, jak wolisz nazwać
print("Model AV zapisano do pliku: model_av.h5")
