from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('model.joblib')
char_encoder = joblib.load('character_encode')
universe_encoder = joblib.load('universe_encode')
abilities_encoder = joblib.load('abilities_encode')
weakness_encoder = joblib.load('weaknes_encode')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        character = request.form['Character']
        universe = request.form['Universe']
        strength = int(request.form['Strength'])
        speed = int(request.form['Speed'])
        intelligence = int(request.form['Intelligence'])
        special_abilities = request.form['SpecialAbilities']
        weaknesses = request.form['Weaknesses']

        # Encode the categorical inputs
        character_encoded = char_encoder.transform([character])[0]
        universe_encoded = universe_encoder.transform([universe])[0]
        special_abilities_encoded = abilities_encoder.transform([special_abilities])[0]
        weaknesses_encoded = weakness_encoder.transform([weaknesses])[0]

        # Create a DataFrame for prediction
        input_data = pd.DataFrame({
            'Character': [character_encoded],
            'Universe': [universe_encoded],
            'Strength': [strength],
            'Speed': [speed],
            'Intelligence': [intelligence],
            'SpecialAbilities': [special_abilities_encoded],
            'Weaknesses': [weaknesses_encoded]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]
        outcome = 'Win' if prediction == 1 else 'Lose'

        return render_template('index.html', prediction_text=f'Battle Outcome: {outcome}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
