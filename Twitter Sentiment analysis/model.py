''' Steps -->
1. Data set import 
2. Data set cleansing and then vectorization
3. Model selection and data dividing
4. Model Training
5. Predicting and Testing the model
'''
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib #to save the trained model
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Data Files/less tweets.csv', encoding="ISO-8859-1")
print(dataset.head)

y = dataset.iloc[:, 0].values

corpus = [] # where I will store the cleaned data
print("cleaning data...")
for i in range(0, len(dataset)):
  review = re.sub('[^a-zA-Z?!]', ' ', dataset['tweets'][i])
  review = review.lower()
  review = review.split()
  #Stemming the data
  stemmer = PorterStemmer()
  review = [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

print("Tweets Cleaned !!")

print("Converting tweets to MOF")
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()

print(x)
joblib.dump(cv, 'count_vectorizer.pkl')

print("splitting...")
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

print("Training the model...")
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
print("Training complete")

print("saving the model...")
joblib.dump(classifier, 'FinalLogisticModel.pkl')
print("model saved successfully")

y_pred = classifier.predict(x_test)
print(y_pred)
print(len(y_pred))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(y_pred)