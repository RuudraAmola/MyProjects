from flask import Flask,render_template,request
import pickle
import numpy as np

# deserializing the .pkl files in read binary mode using the load() function
cv = pickle.load(open("model/cv.pkl","rb"))
mnb = pickle.load(open("model/clf.pkl","rb"))

# creating an instance 'app' to set current python module ie. main.py to the Flask framework
# Flask takes __name__ which represents the current module as an argument
app = Flask(__name__)

@app.route('/')
def webpage():
    return render_template('testpage.html')
# render_template() method calls a template from the template folder specified in the argument

@app.route('/predict_model',methods=['post'])
def predict_model():
    email = request.form.get('email')
# request.form.get() is a flask method which here, sends the entered email on the webpage 
# to the predict_model() function by storing it in the variable 'email'

    # predict for spam
    x_input = cv.transform([email]).toarray()
    y_pred = mnb.predict(x_input)   
    # y_pred is the data predicted by the model from the independent var x_input

    if y_pred[0] == 0:
        response = -1    # indicator for non-spam mail
    else:
        response = 1     # indicator for a spam mail

# using render_template() again we are calling the html template along with the response ie. set above (-1,1) 
    return render_template('testpage.html', response=response)
    
# to run main.py in debugger mode only if it is directly running by python & not imported
if __name__ == "__main__":
    app.run(debug=True)