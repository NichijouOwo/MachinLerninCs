import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, sep=';')

    # Eliminar columnas innecesarias
    columnas_a_eliminar = ['AbnormalMatch', 'TravelledDistance', 'TimeAlive']
    df = df.drop(columnas_a_eliminar, axis=1)

    # Filtrar rondas
    df = df[df['RoundId'] <= 30]

    # Mapear equipos en rondas y partidas
    for partida in df['MatchId'].unique():
        equipo_1 = df[(df['MatchId'] == partida) & (df['InternalTeamId'] == 1) & (df['RoundId'] == 1)]['Team'].values[0]
        equipo_2 = df[(df['MatchId'] == partida) & (df['InternalTeamId'] == 2) & (df['RoundId'] == 1)]['Team'].values[0]

        for ronda in range(1, 31):
            if ronda < 16:
                df.loc[(df['MatchId'] == partida) & (df['RoundId'] == ronda) & (df['InternalTeamId'] == 1), 'Team'] = equipo_1
                df.loc[(df['MatchId'] == partida) & (df['RoundId'] == ronda) & (df['InternalTeamId'] == 2), 'Team'] = equipo_2
            else:
                df.loc[(df['MatchId'] == partida) & (df['RoundId'] == ronda) & (df['InternalTeamId'] == 1), 'Team'] = equipo_2
                df.loc[(df['MatchId'] == partida) & (df['RoundId'] == ronda) & (df['InternalTeamId'] == 2), 'Team'] = equipo_1

    df['FirstKillTime'] = df['FirstKillTime'].astype(bool)
    df.loc[df['FirstKillTime'] & (df['RoundKills'].isnull() | (df['RoundKills'] == 0)), 'RoundKills'] += 1

    # Crear un diccionario de mapeo
    map_dict = {
        'de_inferno': 1,
        'de_nuke': 2,
        'de_mirage': 3,
        'de_dust2': 4
    }

    # Reemplazar los valores de la columna 'Map'
    df['Map'] = df['Map'].replace(map_dict)

    # Eliminar filas con valores NaN en la columna 'MatchWinner'
    df = df.dropna(subset=['MatchWinner'])

    # Convertir 'MatchWinner' a valores numéricos (0 para 'False', 1 para 'True')
    df['MatchWinner'] = df['MatchWinner'].astype(int)

    # Convertir valores booleanos a enteros
    df['RoundWinner'] = df['RoundWinner'].astype(bool).astype(int)

    return df

def save_data(df, pickle_path):
    df.to_pickle(pickle_path)

if __name__ == "__main__":
    filepath = 'dataset/Anexo_ET_demo_round_traces.csv'
    pickle_path = 'checkpoints/df.pkl'
    df = load_and_clean_data(filepath)
    save_data(df, pickle_path)
