import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np 
import pandas as pd 


app = Flask(__name__)
##load the model 
mlp_model = pickle.load(open('mlp.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')

def home():
    return render_template('home.html')


@app.route('/predict_api',methods = ['POST'])

def predict_api(): 
    
    data = request.json['data']  
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output = mlp_model.predict(data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods =['POST'])


def predict():
    
    data =[float(x) for x in request.form.values()]
    output = mlp_model.predict(np.array(data).reshape(1,-1))

    return render_template("home.html", prediction_text = "The sales forcast is {} ".format(output))

    
if __name__=="__main__":

        app.run(debug = True, host  = "0.0.0.0", port = 8080)
        
