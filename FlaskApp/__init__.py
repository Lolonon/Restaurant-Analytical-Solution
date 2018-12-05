from flask import Flask, render_template, request, jsonify, url_for
from flask_bootstrap import Bootstrap 
import sys
import pickle
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from content_management import Content
from recommendation import find_n_most_similar_customer, find_most_popular_restaurants_rated_by_similar_customers, generate_n_most_similar_restaurant
from user_rating_prediction import get_metrics, tfidf
#import recommendation
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from IPython.display import HTML
from surprise import Reader, Dataset, SVD, evaluate
from surprise import KNNBasic
from surprise.accuracy import rmse
from surprise import dump

from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords

TOPIC_DICT = Content()

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def homepage():
    return render_template("home.html")

@app.route('/dashboard/')
def dashboard():
    return render_template("dashboard.html", TOPIC_DICT = TOPIC_DICT)

@app.route('/restaurant_rec/')
def rest_rec():
    return render_template('rec_index.html')

@app.route('/restaurant_rec_predict', methods=['POST'])
def restaurant_rec_predict():
    with open('./data/customer_similarity_computation_data.pkl','rb') as file:
        df_matrix3=pickle.load(file)

    with open('./data/restaurant_similarity_computation_matrix_data.pkl','rb') as file:
        df_restaurant_matrix=pickle.load(file)

    with open('./data/restaurant_to_display_QC_data.pkl','rb') as file:
        df_business_QC=pickle.load(file)
    
    if request.method == 'POST':
	    user_reg = int(request.form['namequery'])

    #user_reg=1
    restaurant_baseline=find_most_popular_restaurants_rated_by_similar_customers(find_n_most_similar_customer(user_reg,start_col='Kazu',end_col='La Folie',n_of_similar_user=10))
    restaurant_rec=generate_n_most_similar_restaurant(find_most_popular_restaurants_rated_by_similar_customers(find_n_most_similar_customer(user_reg,start_col='Kazu',end_col='La Folie',n_of_similar_user=10)))
    
    rest_baseline=df_business_QC[df_business_QC.name.isin(restaurant_baseline)][['name','attribute','categories_clean']]
    rest_rec=df_business_QC.loc[restaurant_rec,:][['name','attribute','categories_clean']]

    base_word_list=(df_business_QC[df_business_QC.name.isin(restaurant_baseline)]['attribute']+df_business_QC[df_business_QC.name.isin(restaurant_baseline)]['categories_clean']).sum().replace('[','').replace(']','').replace(' ','').split("'")
    base_word_list=[x for x in base_word_list if x !=',']
    base_word_list=' '.join(base_word_list).split()
    base_word_list=list(np.unique(base_word_list))

    rec_word_list_0=(df_business_QC[df_business_QC.index.isin(restaurant_rec)]['attribute'].loc[restaurant_rec[0]]+df_business_QC[df_business_QC.index.isin(restaurant_rec)]['categories_clean'].loc[restaurant_rec[0]]).replace('[','').replace(']','').replace(' ','').split("'")
    rec_word_list_0=[x for x in rec_word_list_0 if x !=',']
    rec_word_list_0=' '.join(rec_word_list_0).split()
    rec_word_list_1=(df_business_QC[df_business_QC.index.isin(restaurant_rec)]['attribute'].loc[restaurant_rec[1]]+df_business_QC[df_business_QC.index.isin(restaurant_rec)]['categories_clean'].loc[restaurant_rec[1]]).replace('[','').replace(']','').replace(' ','').split("'")
    rec_word_list_1=[x for x in rec_word_list_1 if x !=',']
    rec_word_list_1=' '.join(rec_word_list_1).split()
    rec_word_list_2=(df_business_QC[df_business_QC.index.isin(restaurant_rec)]['attribute'].loc[restaurant_rec[2]]+df_business_QC[df_business_QC.index.isin(restaurant_rec)]['categories_clean'].loc[restaurant_rec[2]]).replace('[','').replace(']','').replace(' ','').split("'")
    rec_word_list_2=[x for x in rec_word_list_2 if x !=',']
    rec_word_list_2=' '.join(rec_word_list_2).split()
    word_cloud_list=[x for x in rec_word_list_2 if x in base_word_list]+[x for x in rec_word_list_0 if x in base_word_list]+[x for x in rec_word_list_1 if x in base_word_list]
    STOPWORDS
    stopwords = set(STOPWORDS)
    word_text=[x for x in word_cloud_list if x not in stopwords]
    word_freqs = Counter(word_text)
    word_freqs = dict(word_freqs)

    predictions_svd, algo_svd = dump.load('./models/dump_SVD')
    df_temp=pd.DataFrame()
    for rest in restaurant_rec:
        df_temp=df_temp.append([list(algo_svd.predict(uid=1, iid=rest, r_ui=0, verbose=True))])
    df_temp.columns=['uid', 'rid', 'rui', 'est', 'details']
    rest_rec['current customer']=list(df_temp['uid'])
    rest_rec['customer to restaurant rating prediction']=list(df_temp['est'])

    word_freqs_js = []

    for key,value in word_freqs.items():
        temp = {"text": key, "size": value}
        word_freqs_js.append(temp)
    max_freq = max(word_freqs.values())
    return  render_template('rec_predict.html', user_id=user_reg, rest_base=list(rest_baseline['name']), rest_rec=list(rest_rec['name']), word_freqs=word_freqs_js, max_freq=max_freq,  table_baseline=HTML(rest_baseline.to_html(classes='table')), table_rec=HTML(rest_rec.to_html(classes='table')))

@app.route('/restaurant_biz/')
def rest_biz():
    return render_template('biz_index.html')

@app.route('/restaurant_biz_predict/', methods=['POST'])
def restaurant_biz_predict():
    with open('./data/restaurant_rating_avg_X.pkl','rb') as file:
        biz_data=pickle.load(file)
    with open('./data/restaurant_rating_avg_to_display.pkl','rb') as file:
        df_biz=pickle.load(file)
    #with open('./data/user_rating_above_below_avg_prediction_X.pkl','rb') as file:
        #user_review_matrix=pickle.load(file)

    biz_avg_rating_model=joblib.load('./models/restaurant_rating_avg_prediction_model.pkl')
    if request.method == 'POST':
	    rest_reg = int(request.form['namequery'])
    avg_rest_rating_prediction=biz_avg_rating_model.predict(biz_data)[rest_reg]
    
    return render_template('biz_predict.html',prediction = avg_rest_rating_prediction,name = df_biz['name'][rest_reg], rest_id=rest_reg,actual_rating=df_biz['rating'][rest_reg],error=np.abs(avg_rest_rating_prediction-df_biz['rating'][rest_reg]))

@app.route('/restaurant_closure/')
def rest_closure():
    return render_template('restaurant_closure_index.html')

@app.route('/restaurant_closure_predict/', methods=['POST'])
def restaurant_closure_predict():
    with open('./data/restaurant_closure_X.pkl','rb') as file:
        closure_data=pickle.load(file)
    with open('./data/restaurant_closure_to_display.pkl','rb') as file:
        df_closure=pickle.load(file)

    rest_closure_model=joblib.load('./models/restaurant_closure_prediction_model.pkl')
    if request.method == 'POST':
	    rest_reg = int(request.form['namequery'])
    rest_closure_prediction=rest_closure_model.predict(closure_data)[rest_reg]
    
    return render_template('restaurant_closure_predict.html',prediction = rest_closure_prediction,name = df_closure['name'][rest_reg], rest_id=rest_reg)


if __name__ == "__main__":
    app.run()