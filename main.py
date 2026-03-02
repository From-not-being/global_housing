import pandas as pd
from sklearn.ensemble import RandomForestRegressor

global_housing_path = "/content/drive/MyDrive/Colab Notebooks/global_housing.csv"
df_housing = pd.read_csv(global_housing_path)

display(df_housing.head(3))
print(df_housing.columns)

# Pfad zu deiner global housing CSV
path = "/content/drive/MyDrive/Colab Notebooks/global_housing.csv"

def makro_korrelation_housing(file_path):
    print("Lade makroökonomische Daten...\n")
    df = pd.read_csv(file_path)

    # --- 1. SPALTENNAMEN (Bitte exakt an deine CSV anpassen!) ---
    features = ['Mortgage Rate (%)', 'Inflation Rate (%)', 'Population Growth (%)']
    target = 'House Price Index'

    # Leere Felder (NaN) entfernen, sonst stürzt die Mathematik ab
    df = df.dropna(subset=features + [target])

    # --- 2. DIE KLASSISCHE KORRELATION (Gibt es einen direkten Zusammenhang?) ---
    print("=== 1. Lineare Korrelation (Pearson) ===")
    print("Werte nahe +1.0 = Preis steigt mit; Werte nahe -1.0 = Preis fällt; 0 = Kein Zusammenhang\n")

    # Wir korrelieren nur unsere ausgewählten Spalten miteinander
    korrelations_matrix = df[features + [target]].corr()

    # Wir schauen uns nur die Spalte an, die zeigt, wie alles mit dem Preis zusammenhängt
    preis_korrelation = korrelations_matrix[target].drop(target)
    for feature, wert in preis_korrelation.items():
        richtung = "zieht Preis HOCH" if wert > 0 else "drückt Preis RUNTER"
        print(f"-> {feature}: {wert:+.2f} ({richtung})")

    # --- 3. DIE ABSOLUTE WIRKUNGSKRAFT (Scikit-Learn Random Forest) ---
    print("\n=== 2. KI-Einfluss-Analyse (Scikit-Learn Feature Importance) ===")
    print("Welcher Faktor hat das meiste Gewicht bei der Preisbildung?\n")

    X = df[features]
    y = df[target]

    # Modell trainieren (ohne Train/Test-Split, da wir hier nur die Gewichtung analysieren)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = model.feature_importances_
    for feature, imp in zip(features, importances):
        print(f"-> {feature}: {imp * 100:.2f} % Gesamt-Einfluss")

# Skript starten
makro_korrelation_housing(path)
