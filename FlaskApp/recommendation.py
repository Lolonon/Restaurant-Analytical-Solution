import pandas as pd
import numpy as np
import json
import operator
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
############################################################
from surprise import Reader, Dataset, SVD, evaluate
from surprise import KNNBasic
from surprise.accuracy import rmse
from surprise import dump
############################################################
from wordcloud import WordCloud, STOPWORDS
import pickle

with open('./data/customer_similarity_computation_data.pkl','rb') as file:
    df_matrix3=pickle.load(file)

with open('./data/restaurant_similarity_computation_matrix_data.pkl','rb') as file:
    df_restaurant_matrix=pickle.load(file)

with open('./data/restaurant_to_display_QC_data.pkl','rb') as file:
    df_business_QC=pickle.load(file)

def find_n_most_similar_customer(user_id,start_col='Kazu',end_col='La Folie',n_of_similar_user=10):
    top_n_similar_user_id=[]
    top_n_similar_user_similarity=[]
    user_similarity=cosine_similarity(df_matrix3.loc[:,start_col:end_col],df_matrix3.loc[:,start_col:end_col])
    top_n_similar_user_id=np.argsort(user_similarity)[user_id][-(n_of_similar_user+1):-1]
    for similar_user_id in top_n_similar_user_id:
        top_n_similar_user_similarity.append(user_similarity[user_id][similar_user_id])
    
    #print('Customer%d has %d similar users who are:'%(user_id,n_of_similar_user))
    #print(top_n_similar_user_id)
    #print('The similarity to customer%d are:'%user_id)
    #print(top_n_similar_user_similarity)
    #print('-'*110)
    similar_dict=dict(zip(top_n_similar_user_id,top_n_similar_user_similarity))
    #print(similar_dict)    
    return similar_dict

def find_most_popular_restaurants_rated_by_similar_customers(user_list):
    temp_keys=[]
    for user in user_list:
        temp_keys.extend(list(df_matrix3.restaurant_dictionary[user].keys()))
    #print(temp_keys)
    c = Counter(temp_keys)
    dict_c=dict(c.most_common(10))#10 most commonly visited restaurants
    #print (dict(c.most_common(10)))
    
    restaurant_rate_dict={}
    for item in np.unique(temp_keys):# Calculate averaged rating of unique restaurants by similar customers
        temp_restaurant_rates=[]
        for user in user_list:#find all users who visited same restaurants and get the mean rating 
            if item in df_matrix3.restaurant_dictionary[user].keys():
                temp_restaurant_rates.append(float(df_matrix3.restaurant_dictionary[user][item]))
        restaurant_rate_dict[item]=np.mean(temp_restaurant_rates)
    #print(restaurant_rate_dict)   
    rest_list=[]
    for key in dict_c.keys():
        if key in dict(sorted(restaurant_rate_dict.items(), key=operator.itemgetter(1),reverse=True)[:round(len(np.unique(temp_keys))/3)]).keys():
            rest_list.append(key)
    #print('Those customers like you are highly likely choose restaurant:')
    #print('Recommendation Base:' )
    #print(rest_list)
    return rest_list

def generate_n_most_similar_restaurant(rest_list,start_col='review_count',end_col='wings',n_of_similar_restaurant=10,n_of_recommended_restaurant=3):
    top_n_similar_user_rest_preference_id=[]
    for restaurant in rest_list:
        top_n_similar_user_rest_preference_id.extend(df_restaurant_matrix[df_restaurant_matrix.name==restaurant].index)
    top_n_similar_user_rest_preference_id=list(np.unique(top_n_similar_user_rest_preference_id))#Get similar customers' restaurants index preference
    #print(top_n_similar_user_rest_preference_id)
    
    df_ref=pd.DataFrame(columns=df_restaurant_matrix.loc[:,start_col:].columns)#temp avg scored user_preference_vector
    df_ref.loc[0,:]=df_restaurant_matrix.loc[top_n_similar_user_rest_preference_id,start_col:].mean()
    
    restaurant_similarity=cosine_similarity(df_ref,df_restaurant_matrix.loc[:,start_col:]) #compute restaurants similarity
    
    #recommended_rest_list=[]
    #for rest_index in top_n_similar_user_rest_preference_id:
    top_n_similar_rest_id=np.argsort(restaurant_similarity)[0][-(n_of_similar_restaurant+1):-1]
    top_3_recomend_rest=list(df_business_QC.loc[top_n_similar_rest_id,'stars'].sort_values(ascending=False)[:n_of_recommended_restaurant].index)
    #recommended_rest_list.extend(top_3_recomend_rest)
    #print('Recommended 3 Top Restaurants ID: '+ str(top_3_recomend_rest))
    rest_name_3_top=[]
    for ind in top_3_recomend_rest:
        rest_name_3_top.append(df_business_QC.loc[ind,'name'])
    #print('Recommended 3 Top Restaurants Name: '+ str(rest_name_3_top))
    #print(list(np.unique(top_3_recomend_rest)))
    return top_3_recomend_rest