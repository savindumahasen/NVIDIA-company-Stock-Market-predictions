from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle


app=Flask(__name__)
## Load the model
regression_model=pickle.load(open('model.pkl','rb'))
scaling=pickle.load(open('scaler.pkl','rb'))


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/predictions", methods=["POST"])
def predictions():
    data=[float(x) for x in request.form.values()]
    print(data)
    final_input=scaling.transform(np.array(data).reshape(1,-1))
    print(final_input)
    predictions=regression_model.predict(final_input)[0]
    #return jsonify({'Result':predictions})
    return render_template('home.html',prediction_text="Stock market price is  {}".format(predictions)+" USD")





if __name__=="__main__":
    app.run(debug=True, port=5000)