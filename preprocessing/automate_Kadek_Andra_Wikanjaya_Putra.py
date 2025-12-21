import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def preprocessing(df, target_col, output_path):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_final = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=df.index
    )

    df_final = pd.concat([X_final, y], axis=1)
    df_final.to_csv(output_path, index=False)

    return df_final


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "diabetes_raw" / "diabetes.csv"
    OUTPUT_PATH = BASE_DIR.parent / "diabetes_preprocessing.csv"

    df = pd.read_csv(DATA_PATH)

    preprocessing(
        df,
        target_col="Outcome",
        output_path=OUTPUT_PATH
    )

