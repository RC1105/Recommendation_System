from flask import Flask
from flask import request
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv("one.csv")
df.head()

one=[]
two=[]
for l in df.values:
  one.append(l[0])
  two.append([l[2],l[3]])

item_features=dict(zip(one,two))
#print(item_features)

feature_matrix = np.array([item_features[item] for item in item_features])
def calculate_item_similarity(features):
    item_similarity = cosine_similarity(features)
    return item_similarity
def generate_similar_items(item_name, features, item_similarity, num_recommendations=5):
    item_idx = list(item_features.keys()).index(item_name)
    item_scores = item_similarity[item_idx]
    recommended_indices = np.argsort(item_scores)[::-1][:num_recommendations]
    return [list(item_features.keys())[idx] for idx in recommended_indices]


app=Flask(__name__)

@app.route('/post', methods=["POST"])
def func():
    input_item=request.form["value"]
    item_similarity = calculate_item_similarity(feature_matrix)
    recommended_items = generate_similar_items(input_item, feature_matrix, item_similarity)
    lsk=[]
    #print("Items similar to '{}':".format(input_item))
    for item in recommended_items:
        lsk.append(item)
    return lsk


if __name__ =="__main__":
    app.run(host='0.0.0.0')