from flask import Flask, request,render_template
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
with open('model.pkl','rb') as file:
    model = pickle.load(file)

app = Flask(__name__,static_folder='static')
@app.route("/")

def index():
    return render_template('index.html')
@app.route("/Titanic", methods=['POST','GET'])

def Survival():
    
    
    Pclass = int(request.form['Pclass'])
    Sex = int(request.form['Sex'])
    Age = float(request.form['Age'])
    Sibsp = int(request.form['SibSp'])
    Parch = int(request.form['Parch'])
    Fare = float(request.form['Fare'])
    Embarked = float(request.form['Embarked'])
    
    ss = model.predict([[Pclass,Sex,Age,Sibsp,Parch,Fare,Embarked]])
    
    
    
    
    
    
    
    
    
    return render_template('index.html',Survived = ss)
if __name__ ==('__main__'):
    app.run(debug=True)