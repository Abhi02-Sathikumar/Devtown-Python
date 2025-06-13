from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f]) for f in [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]]
        
        final_input = np.array([features])
        prediction = model.predict(final_input)[0]

        return render_template('index.html', prediction_text=f' Predicted House Price: ${prediction:.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f' Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
