import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load ML model from models folder
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'iris_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

species = ['Setosa', 'Versicolor', 'Virginica']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(f)) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    except (TypeError, ValueError):
        return render_template('index.html', prediction_text="Please enter valid numbers.")

    data = np.array(features).reshape(1, -1)
    pred = model.predict(data)[0]
    prediction_text = f"Predicted Iris species: {species[pred]}"
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))   # Heroku assigns PORT; fallback to 5000 locally
    app.run(host='0.0.0.0', port=port, debug=True)