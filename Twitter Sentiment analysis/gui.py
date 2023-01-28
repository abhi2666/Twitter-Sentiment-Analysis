import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import Toplevel
from tkinter import Label
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer
#Working Fucking Fine



root = tk.Tk()

root.title('TWISENT')
root.geometry('800x500')
root.minsize(800, 500)
root.maxsize(800, 500)

def feedandPredict():
    newWindow = Toplevel(root)
    ##now feeding the data to predict the result
    classifier = joblib.load('FinalLogisticModel.pkl')
    print("Predicting the result-->")
    # print("printing the predicted output...")
    cv = joblib.load('count_vectorizer.pkl')
    ## cleaning the data first
    cleaned = []
    for i in range(0, len(data)):
        review = re.sub(r'[\U0001f600-\U0001f650]', '', data['tweets'][i])
        review = re.sub(r'@\w+', '', review)
        review = re.sub('[^a-zA-Z]', ' ', data['tweets'][i])
        cleaned.append(review)

    print("printing cleaned data -->")
    print(cleaned)
    x = cv.transform(cleaned).toarray()
    result = classifier.predict(x)
    print(result)
    # getting positive and negative 
    positive = 0
    negative = 0
    for i in result:
        if(i == 0):
            negative = negative + 1
        elif(i == 4):
            positive = positive + 1

    posPer = (positive/len(result))*100
    negPer = (negative/len(result))*100
    Label(newWindow, text="Positive tweets %"+str(posPer)).pack()
    Label(newWindow, text="Negative tweets %"+str(negPer)).pack()
    

def browse_file():
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a file", filetypes = (("CSV files", "*.csv"), ("all files", "*.*")))
    print(filename)
    # Use pandas to read the selected csv file and store it in a DataFrame
    global data
    data = pd.read_csv(filename, encoding="ISO-8859-1")

manual = Label( text='Twitter Sentiment Analyzer', font= ('Times',20, "bold"), padx = 5, pady = 10)
manual.pack(pady=7)
style = ttk.Style()
style.theme_use("clam") # Using clam theme

browse_button = ttk.Button(root, text ="Browse File", command = browse_file)
browse_button.pack(pady=20, padx=20)
#prediction button
browse_button = ttk.Button(root, text ="Predict", command = feedandPredict)
browse_button.pack(pady=20, padx=20)

root.mainloop()