import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Cargar el DataFrame desde el archivo pickle
def load_data(pickle_path):
    with open(pickle_path, 'rb') as file:
        df = pickle.load(file)
    return df

# Preparar los datos
def prepare_data(df):
    df_rondaspormapa = df.copy()

    # Crear nuevas columnas en el DataFrame
    df_rondaspormapa['Map_1'] = df_rondaspormapa['Map'].apply(lambda x: 1 if x == 1 else 0)
    df_rondaspormapa['Map_2'] = df_rondaspormapa['Map'].apply(lambda x: 1 if x == 2 else 0)
    df_rondaspormapa['Map_3'] = df_rondaspormapa['Map'].apply(lambda x: 1 if x == 3 else 0)
    df_rondaspormapa['Map_4'] = df_rondaspormapa['Map'].apply(lambda x: 1 if x == 4 else 0)
    df_rondaspormapa['Team_CounterTerrorist'] = df_rondaspormapa['Team'].apply(lambda x: 1 if x == 'CounterTerrorist' else 0)
    df_rondaspormapa['Team_Terrorist'] = df_rondaspormapa['Team'].apply(lambda x: 1 if x == 'Terrorist' else 0)

    X = df_rondaspormapa[['RoundKills', 'RoundAssists', 'RoundHeadshots', 'RoundFlankKills',
                          'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
                          'Map_1', 'Map_2', 'Map_3', 'Map_4',
                          'Team_CounterTerrorist', 'Team_Terrorist']]
    y = df_rondaspormapa['RoundWinner']
    
    return X, y

def train_and_evaluate_model(df):
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = make_pipeline(StandardScaler(), LogisticRegression())
    parameters = {'logisticregression__fit_intercept': [True, False]}
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluar el modelo
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

    return best_model

if __name__ == "__main__":
    pickle_path = 'checkpoints/df.pkl'
    model_path = 'checkpoints/linealmodel.pkl'
    scaler_path = 'checkpoints/scaler.pkl'
    
    df = load_data(pickle_path)
    model = train_and_evaluate_model(df)

    # Guardar el modelo y el scaler
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(model.named_steps['standardscaler'], scaler_file)
