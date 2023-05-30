from flask import Flask, render_template, request
import pickle
import pandas as pd

app=Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    age = float(request.form.get("age"))
    sex = float(request.form.get('sex'))
    cp = float(request.form.get('cp'))
    trestbps = float(request.form.get('trestbps'))
    chol = float(request.form.get('chol'))
    fbs = float(request.form.get('fbs'))
    restecg = float(request.form.get('restecg'))
    thalach = float(request.form.get('thalach'))
    exang = float(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = float(request.form.get('slope'))
    ca = float(request.form.get('ca'))
    thal = float(request.form.get('thal'))

    result = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])[0]
    print(result)
    if result==0:
        return render_template("index.html",label=-1)
    else:
        return render_template("index.html",label=1)

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ =='__main__':
    app.run(debug=True)
