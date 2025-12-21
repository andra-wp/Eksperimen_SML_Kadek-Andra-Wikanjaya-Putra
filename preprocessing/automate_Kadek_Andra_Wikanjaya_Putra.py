import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessing(df, output_path="diabetes_preprocessing.csv"):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    df_final = pd.DataFrame(
        df_scaled,
        columns=df.columns,
        index=df.index
    )

    df_final.to_csv(output_path, index=False)

    return df_final
