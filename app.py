# flask use as backend
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__) #initializing

model=pickle.load(open('model.pkl','rb')) #model loading


@app.route('/')  #url for user input
def hello_world():
    return render_template("index1.html")


@app.route('/predict',methods=['POST']) #url for output
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    
    prediction=model.predict(final)[0]
    if prediction==0.0:
        output="No"
    else:
        output="Yes"
    return render_template('index1.html',pred='Employee attrition is {}'.format(output))
   



if __name__ == '__main__':
    app.run(debug=True)
