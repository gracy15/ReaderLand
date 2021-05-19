import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string
import random
from PIL import Image
import requests
from io import BytesIO



def combine_features(data):
  features=[]
  for i in range(0,data.shape[0]):
    features.append(data['title'][i] + '' +data['author'][i] + '' + data['genre'][i])
  return features



def recom_book(book_id):
  df=pd.read_csv('data.csv',encoding='unicode_escape', error_bad_lines=False)

  df['combined_features']=combine_features(df)

  cm=CountVectorizer().fit_transform(df['combined_features'])

  cs=cosine_similarity(cm)

  scores = list(enumerate(cs[book_id]))

  sorted_scores = sorted(scores, key= lambda x:x[1], reverse=True)
  sorted_scores=sorted_scores[1:]

  j=0
  c=0
  ans=[]
  for item in sorted_scores:
    try:
      book_id2= df[df.id == item[0]]['id'].values[0]
      

      ans.append(book_id2)
      j=j+1
      if j>=5:
        break
    except:
      c+=1
  return ans