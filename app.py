from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your model
model = joblib.load('stress_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    study = float(request.form['study'])
    sleep = float(request.form['sleep'])
    exercise = float(request.form['exercise'])
    screen = int(request.form['screen'])

    # Make prediction
    input_data = np.array([[study, sleep, exercise, screen]])
    prediction = model.predict(input_data)[0]

    # Convert to text
    stress_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    result = stress_map[prediction]

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)