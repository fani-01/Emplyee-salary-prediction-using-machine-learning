%%writefile app.py
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

loan_model = joblib.load('loan_model.pkl')
salary_model = joblib.load('salary_model.pkl')

@app.route('/')
def home():
    return '''
    <h2>Loan and Salary Prediction</h2>
    <form action="/predict" method="post">
        <input name="age" placeholder="Age" /><br>
        <input name="income" placeholder="Income" /><br>
        <button type="submit">Predict</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    income = float(request.form['income'])

    features = [[age, income]]

    loan_pred = loan_model.predict(features)[0]
    salary_pred = salary_model.predict(features)[0]

    return f"Loan Prediction: {loan_pred}, Salary Prediction: {salary_pred}"

if __name__ == '__main__':
    app.run(debug=True)
