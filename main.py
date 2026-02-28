import csv
from sklearn.ensemble import RandomForestRegressor
import numpy as np

housing_path = "/content/drive/MyDrive/Colab Notebooks/global_housing.csv"

def run_housing_robust(file_path):
    X, y = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        actual_columns = reader.fieldnames
        print(f"Gefundene Spalten in der Datei: {actual_columns}")

        # Define the actual columns to use from the CSV
        feature_cols = ['Rent Index', 'Affordability Ratio', 'Mortgage Rate (%)', 'Inflation Rate (%)', 'GDP Growth (%)']
        target_col = 'House Price Index'

        for i, r in enumerate(reader):
            try:
                # Extract features and target using actual column names
                features = [
                    float(r.get(col, 0)) for col in feature_cols
                ]
                target = float(r.get(target_col, 0))

                # Only add data if all required columns were found and converted successfully
                if all(col in r and r[col] for col in feature_cols + [target_col]): # Check for existence and non-empty string
                    X.append(features)
                    y.append(target)
                else:
                    # Skip rows where essential data is missing or empty
                    print(f"Skipping row {i+1} due to missing or empty essential data.")

            except ValueError as e:
                print(f"Skipping row {i+1} due to data conversion error: {e}")
                continue # Skip rows with conversion errors

    if not X:
        print("❌ Fehler: Keine Daten konnten extrahiert werden. Prüfe die Spaltennamen und Daten im CSV!")
        return

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    print("\n✅ Erfolg! Scikit-Learn Modell trainiert.")
    print(f"Anzahl verarbeiteter Datensätze: {len(X)}")

    # Feature Importance
    importances = model.feature_importances_
    features_names = feature_cols # Use the actual feature column names for importance display

    print("\n📊 Wichtigste Preistreiber (für 'House Price Index'):")
    for name, imp in zip(features_names, importances):
        print(f"{name}: {imp:.2%}")

run_housing_robust(housing_path)
