import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
lemmatiser = WordNetLemmatizer()

useless_words = stopwords.words("english")


b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def translate_personality(personality):
    # transform mbti to binary vector
    return [b_Pers[l] for l in personality]

#To show result output for personality prediction
def translate_back(personality):
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

# kullanıcıdan alınan veri düzenleyen fonksiyon
def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
  unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
  list_personality = []
  list_posts = []
  len_data = len(data)
  i=0
  
  for row in data.iterrows():
      # check code working 
      # i+=1
      # if (i % 500 == 0 or i == 1 or i == len_data):
      #     print("%s of %s rows" % (i, len_data))

      #Remove and clean comments
      posts = row[1].posts

      #Remove url links 
      temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

      #Remove Non-words - keep only words
      temp = re.sub("[^a-zA-Z]", " ", temp)

      # Remove spaces > 1
      temp = re.sub(' +', ' ', temp).lower()

      #Remove multiple letter repeating words
      temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

      #Remove stop words
      if remove_stop_words:
          temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
      else:
          temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
          
      #Remove MBTI personality words from posts
      if remove_mbti_profiles:
          for t in unique_type_list:
              temp = temp.replace(t,"")

      # transform mbti to binary vector
      type_labelized = translate_personality(row[1].type) #or use lab_encoder.transform([row[1].type])[0]
      list_personality.append(type_labelized)
      # the cleaned data temp is passed here
      list_posts.append(temp)

  # returns the result
  list_posts = np.array(list_posts)
  list_personality = np.array(list_personality)
  return list_posts, list_personality





app = Flask(__name__)

model = pickle.load(open("mbti_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

@app.route("/predict", methods=["POST"])
def predict():	
    my_data = """ """
    for x in request.form.values(): 							
        my_data += str(x)
        my_data += " "
    mydata = pd.DataFrame(data={'type': ['INTJ'], 'posts': [my_data]})                  
    my_posts, dummy  = pre_process_text(mydata, remove_stop_words=True, remove_mbti_profiles=True) 
    
    cv_pickle = pickle.load(open("cv.pkl", "rb"))
    my_X_cv = cv_pickle.transform(my_posts)
    
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    my_X_tfidf = tfidf.transform(my_X_cv)
    
    prediction = model.predict(my_X_tfidf)

    output = prediction[0]

    types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

    return render_template(
        "index.html", prediction_text="Your type is: {}".format(output)
    )


if __name__ == "__main__":
    app.run(debug=True)
