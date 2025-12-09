import math #moduł z funkcjami matematycznymi
import numpy as np
import pandas as pd
import joblib #biblioteka do zapisywania/ładowania obiektów Pythona, np. skalera (scaler.pkl)
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ------------------------------------
# USTAWIENIA PODSTAWOWE


# Ścieżka do danych (ta sama co w modelach)
DATA_PATH = r"C:\Users\Marta\Desktop\Input_data_cvss\data_word_50.csv"

# Nazwy zapisanych modeli – DOPASOWUJE DO TEGO, CO ZAPISANE Z TYCH PLIKOW GDZIE SIE UCZYLO
MODEL_PATHS = {
    "AV": "model_av.h5",
    "AC": "model_ac.h5",
    "PR": "model_pr.h5",
    "UI": "model_ui.h5",
    "S":  "model_s.h5",
    "C":  "model_c.h5",
    "I":  "model_i.h5",
    "A":  "model_a.h5",
}

# Później te modele poprostu wczytamy

SCALER_PATH = "scaler.pkl"
# scieżka do pliku gdzie jest zapisany skaler AV
N_FEATURES = 57   # tyle cech używałaś
TEST_SIZE = 3179  # tak jak w treningu
RANDOM_STATE = 42


# -----------------------------------------------
# MAPOWANIA KLAS NA LITERKI I WARTOŚCI CVSS 3.1


# Attack Vector (AV): klasy 0–3 → litery + wagi
AV_CLASS_TO_LETTER = {
    0: "P",  # Physical
    1: "L",  # Local
    2: "A",  # Adjacent
    3: "N",  # Network
}
AV_CLASS_TO_WEIGHT = {
    0: 0.20,
    1: 0.55,
    2: 0.62,
    3: 0.85,
}
#z dokumentacji

# Attack Complexity (AC): 0–1
AC_CLASS_TO_LETTER = {
    0: "L",  # Low
    1: "H",  # High
}
AC_CLASS_TO_WEIGHT = {
    0: 0.77,  # Low
    1: 0.44,  # High
}

# Privileges Required (PR): 0–2, wagi zależą od Scope (S)
PR_CLASS_TO_LETTER = {
    0: "N",  # None
    1: "L",  # Low
    2: "H",  # High
}

# User Interaction (UI): 0–1
UI_CLASS_TO_LETTER = {
    0: "N",  # None
    1: "R",  # Required
}
UI_CLASS_TO_WEIGHT = {
    0: 0.85,  # None
    1: 0.62,  # Required
}

# Scope (S): 0–1
S_CLASS_TO_LETTER = {
    0: "U",  # Unchanged
    1: "C",  # Changed
}

# C, I, A (0–2)
CIA_CLASS_TO_LETTER = {
    0: "N",  # None
    1: "L",  # Low
    2: "H",  # High
}
CIA_CLASS_TO_WEIGHT = {
    0: 0.00,
    1: 0.22,
    2: 0.56,
}


# -------------------------------------------------------
# 3. FUNKCJA ZAOKRĄGLANIA CVSS do 1 miejsca

def round_up_1_decimal(x: float) -> float:
    return math.ceil(x * 10.0) / 10.0


# ---------------------------------------------
# 4. FUNKCJA LICZĄCA BASE SCORE CVSS 3.1


def calculate_base_score_from_classes(av_cls, ac_cls, pr_cls, ui_cls, s_cls, c_cls, i_cls, a_cls):
    """
    Wejście: klasy numeryczne (takie jak w danych: 0,1,2,3...)
    Wyjście:
      - base_score (float)
      - słownik z literkami (AV,AC,PR,UI,S,C,I,A)
    """
    # przyjmuje wyniki modeli  i zwraca base score i wektor
    # Literki
    av_letter = AV_CLASS_TO_LETTER.get(av_cls, "?") #bierze z tych moich zapisanych danych literke i przypisuje dla klasy av_cls
    ac_letter = AC_CLASS_TO_LETTER.get(ac_cls, "?")
    pr_letter = PR_CLASS_TO_LETTER.get(pr_cls, "?")
    ui_letter = UI_CLASS_TO_LETTER.get(ui_cls, "?")
    s_letter  = S_CLASS_TO_LETTER.get(s_cls, "?")
    c_letter  = CIA_CLASS_TO_LETTER.get(c_cls, "?")
    i_letter  = CIA_CLASS_TO_LETTER.get(i_cls, "?")
    a_letter  = CIA_CLASS_TO_LETTER.get(a_cls, "?") # znak zaptania na wszelki wypadek jakby klasa była z poza słownika

    # Wagi liczbowe     biore wartości liczbowe, które potem bedą do wzoru
    av = AV_CLASS_TO_WEIGHT[av_cls]#bierze z tych moich zapisanych danych wartosc liczbowa i przypisuje dla klasy av_cls
    ac = AC_CLASS_TO_WEIGHT[ac_cls]
    ui = UI_CLASS_TO_WEIGHT[ui_cls]
    c  = CIA_CLASS_TO_WEIGHT[c_cls]
    i  = CIA_CLASS_TO_WEIGHT[i_cls]
    a  = CIA_CLASS_TO_WEIGHT[a_cls]

    # Wagi PR zależą od Scope (U / C)
    if s_letter == "U":
        # Scope Unchanged
        pr_weights = {
            0: 0.85,  # None
            1: 0.62,  # Low
            2: 0.27,  # High
        }
    else:
        # Scope Changed
        pr_weights = {
            0: 0.85,  # None
            1: 0.68,  # Low
            2: 0.50,  # High
        }

    pr = pr_weights[pr_cls]

    # Exploitability sub-score z dokumentacji CVSS v3.1
    exploitability = 8.22 * av * ac * pr * ui

    # Impact sub-score (Impact) z dokumentacji CVSS v3.1
    impact_sub = 1 - ((1 - c) * (1 - i) * (1 - a))

    if s_letter == "U":
        impact = 6.42 * impact_sub
    else:
        impact = 7.52 * (impact_sub - 0.029) - 3.25 * pow((impact_sub - 0.02), 15)

    # Jeśli nie ma wpływu – score = 0
    if impact <= 0:
        base_score = 0.0 # z dokumnetacji :If the Impact Sub-Score is zero, the Base Score is zero.
    else:
        if s_letter == "U":
            temp = impact + exploitability
        else:
            temp = 1.08 * (impact + exploitability)

        base_score = round_up_1_decimal(min(temp, 10.0))

    letters = {
        "AV": av_letter,
        "AC": ac_letter,
        "PR": pr_letter,
        "UI": ui_letter,
        "S":  s_letter,
        "C":  c_letter,
        "I":  i_letter,
        "A":  a_letter,
    }

    return base_score, letters
  # taki słównik z literkami, funkcja zwraca base score i literki

# ------------------------------
# 5. GŁÓWNY PRZEPŁYW


def main():
    print("=== Ewaluacja Base Score CVSS 3.1 na zbiorze testowym ===\n")

    # 1) Wczytanie danych tak jak w modelach
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
    print("Kształt danych:", df.shape)

    # X – pierwsze 57 kolumn, jak w modelach
    X = df.iloc[:, 0:N_FEATURES]

    # y_base – prawdziwy Base Score CVSS 3.1 ( indeks 65)
    y_base_true = df.iloc[:, 65]

    # 2) Podział train/test – taki sam jak w treningu
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_base_true,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    print("\nRozmiary zbiorów:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # Zapamiętujemy indeksy wierszy testowych, żeby pobrać prawdziwy BSS
    test_indices = X_test.index

    # 3) Skalowanie – ten sam scaler, co w treningu (scaler.pkl)
    print("\nWczytywanie skalera z pliku:", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_test_scaled = scaler.transform(X_test)

#oblib.load("scaler.pkl") – wczytuje skalera nauczonego w pliku
#scaler.transform(X_test) – skaluje dane testowe identycznie jak treningowe
#Dzięki temu dane wejściowe do modeli są w takim samym zakresie, jak przy uczeniu

    # Wczytanie modeli
    print("\nWczytywanie modeli...")
    models = {}  # pusty słownik na modele
    for key, path in MODEL_PATHS.items(): #leci po tym model_av itd
        print(f"  {key}: {path}")
        models[key] = tf.keras.models.load_model(path) #wczytuje model z pliku .h5 i sapisyeane sa one w słowniku
    print("Modele załadowane.\n")

    # Predykcje numeryczne AV, AC, PR, UI, S, C, I, A
    metrics = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"] #lista skladowych wektora
    preds_num = {} # slownik na przewidziane klasy

    for m in metrics:
        model = models[m] #bierzesz odpowiedni model ze słownik
        proba = model.predict(X_test_scaled, verbose=0)   #dla każdego rekordu z testu zwraca wektor prawdopodobieństw
        cls = np.argmax(proba, axis=1)                    # bierzesz indeks największego prawdopodobieństwa:
        preds_num[m] = cls #zapisujesz cały wektor klas (dla wszystkich 3179 próbek)

    preds_num_df = pd.DataFrame(preds_num, index=test_indices) # tworzy tabele z tymi predykcjami

    # Składamy wyniki do jednego DataFrame z oryginalnymi danymi testowymi
    result_df = df.loc[test_indices].copy()

    # Numeryczne predykcje
    for m in metrics:
        result_df[f"{m}_pred_num"] = preds_num_df[m]

    # Liczenie Base Score + literki dla każdej próbki
    base_scores_pred = []
    av_letters = []
    ac_letters = []
    pr_letters = []
    ui_letters = []
    s_letters = []
    c_letters = []
    i_letters = []
    a_letters = []
    #puste klasy i tu bedziemy wrzycac base score te literki itd

    for idx, row in preds_num_df.iterrows():
        av_cls = int(row["AV"])
        ac_cls = int(row["AC"])
        pr_cls = int(row["PR"])
        ui_cls = int(row["UI"])
        s_cls  = int(row["S"])
        c_cls  = int(row["C"])
        i_cls  = int(row["I"])
        a_cls  = int(row["A"])

    #klasa numeryczna przewidziana przez model AV dla tej próbki
        #rzutujesz na zwykłą liczbę całkowitą
        bss_pred, letters = calculate_base_score_from_classes(
            av_cls, ac_cls, pr_cls, ui_cls, s_cls, c_cls, i_cls, a_cls
        )


        base_scores_pred.append(bss_pred)
        av_letters.append(letters["AV"])
        ac_letters.append(letters["AC"])
        pr_letters.append(letters["PR"])
        ui_letters.append(letters["UI"])
        s_letters.append(letters["S"])
        c_letters.append(letters["C"])
        i_letters.append(letters["I"])
        a_letters.append(letters["A"])

    # Dodajemy literki i Base Score do result_df
    result_df["AV_pred"] = av_letters
    result_df["AC_pred"] = ac_letters
    result_df["PR_pred"] = pr_letters
    result_df["UI_pred"] = ui_letters
    result_df["S_pred"]  = s_letters
    result_df["C_pred"]  = c_letters
    result_df["I_pred"]  = i_letters
    result_df["A_pred"]  = a_letters

    result_df["BSS_pred"] = base_scores_pred

    # Prawdziwy Base Score z danych (kolumna 65)
    result_df["BSS_true"] = df.loc[test_indices, 65]

    # z oryginalnej tabeli bierze kolumnę 65 (prawdziwy Base Score), tylko dla wierszy testowych i sapisuje jako BSS_true

    # Błąd bezwzględny
    result_df["BSS_error_abs"] = (result_df["BSS_true"] - result_df["BSS_pred"]).abs()

    #odejmuje przewidziany wynik od prawdziwego
    #.abs() wartoscw bezwzgledna
    #I tworzy kolumne bSS_error_abd - błąd absolutny

    print("\nStatystyki błędu Base Score na zbiorze testowym:")
    print("Średni błąd absolutny:", result_df["BSS_error_abs"].mean())
    print("Mediana błędu absolutnego:", result_df["BSS_error_abs"].median())
    print("Odsetek próbek z błędem <= 0.5:",
          (result_df["BSS_error_abs"] <= 0.5).mean())
    print("Odsetek próbek z błędem <= 1.0:",
          (result_df["BSS_error_abs"] <= 1.0).mean())

    # Zapis wyników do CSV
    output_path = r"C:\Users\Marta\Desktop\Input_data_cvss\base_score_eval_test.csv"
    result_df.to_csv(output_path, index=True)
    print("\nWyniki zapisano do pliku:")
    print(output_path)

    # Pokaż kilka przykładowych wierszy
    print("\nPrzykładowe wiersze (kolumny kluczowe):")
    print(result_df[[
        65,             # oryginalny BSS_CVSSv3.1
        "BSS_true",
        "BSS_pred",
        "BSS_error_abs",
        "AV_pred", "AC_pred", "PR_pred", "UI_pred",
        "S_pred", "C_pred", "I_pred", "A_pred"
    ]].head())


if __name__ == "__main__":
    main()
