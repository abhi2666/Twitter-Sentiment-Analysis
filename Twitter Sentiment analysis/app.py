from flask import Flask, render_template, request
import joblib
import pandas as pd
import re

app = Flask(__name__)

def cleanData(dataset):
    cleaned = []
    for i in range(0, len(dataset)):
        review = re.sub(r'[\U0001f600-\U0001f650]', '', dataset['tweets'][i])
        review = re.sub(r'@\w+', '', review)
        review = re.sub('[^a-zA-Z]', ' ', dataset['tweets'][i])
        cleaned.append(review)

    return cleaned


def getOutput(result):
    #to get the final output
    positive = 0
    negative = 0
    for i in result:
        if(i == 0):
            negative = negative + 1
        elif(i == 4):
            positive = positive + 1

    posPer = (positive/len(result))*100
    negPer = (negative/len(result))*100
    return render_template('result.html', positive = posPer, negative = negPer)

    
@app.route("/", methods=['GET', 'POST'])
def uploadFiles():
    if request.method == 'POST':
        data = request.files['file']
        classifier = joblib.load('FinalLogisticModel.pkl')
        print("Model loaded successfully !")
        cv = joblib.load('count_vectorizer.pkl')
        print(data)
        dataset = pd.read_csv(data, encoding="ISO-8859-1")
        dataset = cleanData(dataset)
        dataset = cv.transform(dataset).toarray()
        print(dataset)
        print(len(dataset))
        result = classifier.predict(dataset)
        
        return getOutput(result)

    elif request.method == 'GET':
        return render_template('index.html')


     
if (__name__ == "__main__"):
     app.run(port = 5000)