from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, SelectField
from wtforms.validators import InputRequired
import numpy as np
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class PredictionForm(FlaskForm):
    round_kills = FloatField('Round Kills', validators=[InputRequired()])
    round_assists = FloatField('Round Assists', validators=[InputRequired()])
    round_headshots = FloatField('Round Headshots', validators=[InputRequired()])
    round_flank_kills = FloatField('Round Flank Kills', validators=[InputRequired()])
    round_starting_equipment_value = FloatField('Round Starting Equipment Value', validators=[InputRequired()])
    team_starting_equipment_value = FloatField('Team Starting Equipment Value', validators=[InputRequired()])
    
    map = SelectField('Map', choices=[('de_inferno', 'de_inferno'), ('de_nuke', 'de_nuke'), ('de_mirage', 'de_mirage'), ('de_dust2', 'de_dust2')], validators=[InputRequired()])
    team = SelectField('Team', choices=[('team_counter_terrorist', 'Counter Terrorist'), ('team_terrorist', 'Terrorist')], validators=[InputRequired()])
    
    submit = SubmitField('Predict')

model_path = 'checkpoints/linealmodel.pkl'
scaler_path = 'checkpoints/scaler.pkl'

with open(model_path, 'rb') as model_file:
    logistic_model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/', methods=['GET', 'POST'])
def home():
    form = PredictionForm()
    if form.validate_on_submit():
        # Obtener datos del formulario
        input_data = [
            form.round_kills.data,
            form.round_assists.data,
            form.round_headshots.data,
            form.round_flank_kills.data,
            form.round_starting_equipment_value.data,
            form.team_starting_equipment_value.data,
        ]
        
        # Convertir mapa y equipo seleccionados en variables dummy
        map_dict = {'de_inferno': 1, 'de_nuke': 2, 'de_mirage': 3, 'de_dust2': 4}
        team_dict = {'team_counter_terrorist': 'CounterTerrorist', 'team_terrorist': 'Terrorist'}
        
        map_value = map_dict[form.map.data]
        team_value = team_dict[form.team.data]
        
        map_dummy = [1 if i == map_value else 0 for i in range(1, 5)]
        team_dummy = [1 if team_value == 'CounterTerrorist' else 0, 1 if team_value == 'Terrorist' else 0]
        
        # Agregar variables dummy a los datos de entrada
        input_data.extend(map_dummy)
        input_data.extend(team_dummy)

        # Asegurarse de que el orden de las variables dummy coincida con el modelo
        input_data = np.array([input_data])
        input_data = scaler.transform(input_data)  # Escalar los datos de entrada

        prediction = logistic_model.predict(input_data)[0]  # Hacer la predicción

        return render_template('result.html', prediction=prediction)
    
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
