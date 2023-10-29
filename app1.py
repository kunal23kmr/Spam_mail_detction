from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict_fun():
    # Open the pickled models using joblib
    with open('spam-mail.pkl', 'rb') as spam_model_file:
        clf = joblib.load(spam_model_file)

    with open('cv.pkl', 'rb') as cv_model_file:
        cv = joblib.load(cv_model_file)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('result.html', prediction=my_prediction[0])


if __name__ == '__main':
    app.run(debug=True)
