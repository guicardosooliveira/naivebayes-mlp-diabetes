import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataframe(path):
    return pd.read_csv(path)

def transform_dataframe(df):
    # Armazena todas as variáveis binárias (SIM ou NÃo) e remove Gender para ser tratado separadamente
    bin_columns = df.columns[df.dtypes == 'object'].tolist()
    bin_columns.remove('Gender')

    #Transforma "Yes -> 1" e "No -> 0"
    df[bin_columns] = df[bin_columns].replace({'Yes': 1, 'No': 0})
    df[bin_columns] = df[bin_columns].infer_objects(copy=False)

    #Trata a coluna Gender (Male -> 1 e Female -> 0)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # 5. Converte a coluna 'class' para 1 = positivo, 0 = negativo
    df['class'] = df['class'].map({'Positive': 1, 'Negative': 0})

    return df

def split_train_test_data(x, y):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42
    )
    return X_train, X_test, y_train, y_test