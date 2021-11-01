from flask import Flask, render_template, request
import pickle
import pandas as pd

with open(f'model/lr.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('main.html')

    if request.method == 'POST':
        volatileAcidity = request.form['volatileAcidity']
        citricAcid = request.form['citricAcid']
        totalSulfurDioxide = request.form['totalSulfurDioxide']
        density = request.form['density']
        sulphates = request.form['sulphates']
        alcohol = request.form['alcohol']

        input_variables = pd.DataFrame([[volatileAcidity, citricAcid, totalSulfurDioxide, density, sulphates, alcohol]],
                                       columns=[
                                           'volatile acidity', 'citric acid', 'total sulfur dioxide', 'density', 'sulfates', 'alcohol'],
                                       dtype="int64")
        # input_variables = [[volatileAcidity, citricAcid,
        #                     totalSulfurDioxide, density, sulphates, alcohol]]
        predictions = model.predict(input_variables)
        # return redirect(url_for('submitForm'))
        return render_template('main.html', result=format(predictions[0], ".4f"))


if __name__ == '__main__':
    app.run(debug=True)
