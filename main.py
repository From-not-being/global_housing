import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Pfad zu deiner global housing CSV
path = "/content/drive/MyDrive/Colab Notebooks/global_housing.csv"

def scikit_housing_analysis_with_print(file_path):
    print("Starte Scikit-Learn Analyse für den Immobilienmarkt...\n")
    
    # 1. Daten laden
    df = pd.read_csv(file_path)
    
    features = ['Mortgage Rate (%)', 'Inflation Rate (%)', 'Population Growth (%)']
    target = 'House Price Index'
    
    # 2. Datenreinigung (NaNs entfernen, damit die Mathematik nicht bricht)
    df = df.dropna(subset=features + [target])
    X, y = df[features], df[target]

    # 3. Modell & Training (Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Feature Importance extrahieren (in Prozent umrechnen)
    importances = model.feature_importances_ * 100

    # --- 5. ERGEBNISSE IN DIE KONSOLE DRUCKEN ---
    print("=== STATISTISCHE GEWICHTUNG (ERGEBNISSE) ===")
    print(f"Anzahl der analysierten Datensätze: {len(df)}")
    print("-" * 45)
    
    # Wir sortieren die Ergebnisse für den Druck (optional, aber schöner)
    results = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    
    for feature, importance in results:
        print(f"-> {feature:<25}: {importance:>6.2f} %")
    
    print("-" * 45)
    print("Analyse abgeschlossen.\n")

    # 6. Visualisierung mit Matplotlib
    plt.figure(figsize=(10, 6))
    colors = ['#2E86C1', '#5DADE2', '#AED6F1'] # Verschiedene Blautöne
    bars = plt.bar(features, importances, color=colors, edgecolor='black')
    
    plt.title('Housing Market: Feature Importance (Scikit-Learn)', fontsize=14)
    plt.ylabel('Bedeutung für den Preis (%)', fontsize=12)
    plt.ylim(0, max(importances) + 10)
    
    # Werte direkt auf die Balken schreiben
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, 
                 f'{yval:.2f}%', ha='center', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Skript starten
scikit_housing_analysis_with_print(path)
