import requests
import time
import json
import hydralit_components as hc
import streamlit as st
import base64
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

# Data dependencies
import pandas as pd
import numpy as np
import model_app
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px
import base64

st.set_page_config(page_title="ThynkData", layout="wide", page_icon=":sparkles:")

news_vectorizer = open("resources/Count_Vectorizer.pkl","rb")
tweet_vect = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

def load_lottieurl(url: str):
    r = requests.get(url)

    if r.status_code != 200:
        return None
    return r.json()    

load_lottie_about = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_1TcivY.json")
load_lottie_background = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qaemdbel.json")
#load_lottie_conclusion = load_lottieurl("https://assets10.lottiefiles.com/private_files/lf30_ndkhjs5v.json")
load_lottie_twitter = load_lottieurl("https://assets10.lottiefiles.com/private_files/lf30_ndkhjs5v.json")
raw = pd.read_csv("resources/train.csv")


# specify the primary menu definition
#Main menu
menu_data = [
    {'id':'Background','icon':"⌚️",'label':"Background"},
    {'id':'About Us','icon': "✍️", 'label':"About Us"},
    {'id':'App Tour','icon': "💻", 'label':"App Tour"},
    {'id':'Data Analysis','icon': "📊", 'label':"Data Analysis"},
    {'id':'Tweet Prediction','icon': "📈", 'label':"Tweet Prediction"},
    {'id':'Improvements','icon': "✍️", 'label':"Improvements"}
]

over_theme = {'txc_inactive': '#FFFFFF','menu_background': '#C21616',}
selected = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='ThynkData',
    hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

col1, col2 = st.columns((2, 1))
img1,img2,img3 = st.columns(3)

if selected == "ThynkData":
    st_lottie(load_lottie_about, speed=1, loop=True, quality="high", width=900, reverse=True)
    #st.write("You selected the home page!")
    #st.write(load_lottie_about)
elif selected == "Background":

    with col1:
        st.title("Background")
        st.write("##")
        st.info("Bio Straw has tasked Thynk Data to create a mechine learning model that is able to classify whether  or not a person belives in climate change based on their novel tweet data")
        st.info ("Bio straw like many other comapnies strive to offer products and services that are enviormentally firiendly and sustainable in line with their values and ideals. With this said Bio straw would like to know how people percive cimate change and whether or not they belive it is a real threat. This information would add to their market research efforts in gauging how their prducts and services may be recieved.")
        st.write("##")
       
    with col2:
        st.write("###") 
        st.write("###") 
        st.write("###")  
        st.write("###")
        st.write("###") 
        st_lottie(load_lottie_background, speed=1, loop=True, width=350,  quality="high", reverse=True)

    with img1:
        image = Image.open(os.path.join("resources/imgs/ThynkData_Dev_Process.jpg"))
        st.image(image,width=700, caption='ThynkData Development Process')
        
    with img2:
        st.write(" ")
        
    with img3:
        st.write(" ")
    

elif selected == "About Us":
    st.title("Our Story")
    st.info("Our CEO, Dr Craig Nyatondo, an internationally published data scientist and at the time computer science lecturer at the University of Stellenbosch, founded ml4africa.com (Machine Learning for Africa) in 2013 after he saw the potential of his field of study to impact communities in South Africa and beyond. One year later Thynk Data with its agile and innovative business model was born as the result of a keen understanding of the predictive analysis needs of governmental stakeholders as well as corporate clients. The integration of theory and praxis lies at the heart of who Thynk Data is. Our data crafters are accomplished engineers, mathematicians and scientists, who are much respected in their respective fields of specialisation.")
    st.subheader("We belive in Purpose before profit")
    st.info("Our purpose is to create value by collaboratively crafting elegant, data-driven solutions for significant problems. To achieve this, we subscribe to the values of Trustworthy Leadership, Collaborative Learning and Creative Craftsmanship. To our clients, Praelexis promises to be Trendsetters, Academically Excellent, Agile and Adaptable and Deeply Immersed.")
    st.subheader("Meet the Team")
    fig_col1, fig_col2,fig_col3,fig_col4,fig_col5 = st.columns(5)
    with fig_col1:
        image_climate = Image.open(os.path.join("resources/imgs/Craig.jpg"))
        image_climate = image_climate.resize((300,300))
        st.image(image_climate, caption='Chief Execitive Officer: Dr Craig Nyatondo ')

    with fig_col2:
        image_climate = Image.open(os.path.join("resources/imgs/Caitlin.jpg"))
        image_climate = image_climate.resize((300,300))
        st.image(image_climate, caption='Chief Infrmation Officer: caitlin Mclaren')

    with fig_col3:
        image_climate = Image.open(os.path.join("resources/imgs/Karabo.jpg"))
        image_climate = image_climate.resize((300,300))
        st.image(image_climate, caption= 'Senior Data Engineer: Karabo Ratona')
    with fig_col4:
        image_climate = Image.open(os.path.join("resources/imgs/Nomonde.jpg"))
        image_climate = image_climate.resize((300,300))
        st.image(image_climate, caption='Senior Interface Developer: Nomonde Mraqisa')
    with fig_col5:
        image_climate = Image.open(os.path.join("resources/imgs/Mamtie.jpg"))
        image_climate = image_climate.resize((300,300))
        st.image(image_climate, caption='Lead full stack Engineer: Mamutele Phosa')
   
elif selected == "App Tour":
    selected = option_menu(
		menu_title="Main Menu",
		options=["About us", "Background", "Data analysis", "Tweet Prediction", "Improvements"],
		icons=["people-fill", "book-half", "bar-chart-line-fill", "graph-up","book"],
		menu_icon="cast",
		default_index=0,
		orientation="horizontal",
	)
    st.title("Good Day Bio straw :wave:")
    st.title("Welcome to Thynk Data's App :smile:")
    st.write("" * 34)

elif selected  == "Data Analysis":
    st.subheader("Lets analyze our data")
    upload_file = model_app.upload_file()
    if upload_file is not None:
        raw = pd.read_csv(upload_file )
        with st.expander("File Data"):
            with hc.HyLoader('Loading....',hc.Loaders.standard_loaders,index=[2,2,2,2], primary_color='#C21616', height='100'):
                st.write(raw) # will write the df to the page
        with st.expander("Wordmap of the Uploaded Data"):
            with hc.HyLoader('',hc.Loaders.pulse_bars, primary_color='#C21616', height='100'):
                testing_wordMap = model_app.word_map(raw)
  
        st.markdown("### Tweet distribution")
        sentiment = raw['sentiment'].value_counts()
        sentiment = pd.DataFrame({'Sentiment':sentiment.index, 'Tweets':sentiment.values})

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            fig = fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets',   color_continuous_scale='peach' , height= 500)
            st.plotly_chart(fig)
        with fig_col2:
            fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment', color_discrete_sequence=px.colors.sequential.RdBu)
			
            st.plotly_chart(fig)
elif selected  == "Tweet Prediction":
    st.info("In this section we are going to be classifying tweets using the models listed below. You can only select one model at a time when classifying a tweet. Once your tweet has been typed proceed to click on the clasfy button below to see which category your tweet falls under.")
    model = st.radio(
    	"*Select a model to classifiy your tweet*",
    		('Logistic_regression','Naive_Bayes','Linear_Support_Vector','Random Forest','K-Nearest Neighbour' ))
		# Creating a text box for user input

    if model == 'Random Forest' :
        st.success("A random forest model is a form of an ensemble model. It essentially works by creating a number of decison tree models and combines their predicitions to produce a more accurate predition than 1 single model would." )
        tweet_text = st.text_area("Type a tweet")
        tweet_text = model_app.cleaning_text(tweet_text)
        if st.button("Classify"):
            # Transforming user input with vectorizer	
            rfc_text = tweet_vect.transform([tweet_text]).toarray()
			# Load your randomfc_model.pkl file 
            predictor = joblib.load(open(os.path.join("resources/Random_Forest.pkl"),"rb"))
            prediction = predictor.predict(rfc_text)
            results = model_app.classify_desc(format(prediction))
            # When model has successfully run, will print prediction
            st.success("Your tweet is classified as: {} ".format(results) )

    if model == 'Logistic_regression':
        st.success("Logistic Regression model in simple terms the logistic regression model is similar to the linear regression model, in that it assumes a base linear relationship between the predictors and response variable. The main differing factor is that because it is a classification model it does not predict an exact value or in this case a class, but it predicts the probability of a specific class ie 0 or 1." )
        tweet_text = st.text_area("Type a tweet")
        tweet_text = model_app.cleaning_text(tweet_text)
        if st.button("Classify"):
			# Transforming user input with vectorizer
            vect_text = tweet_vect.transform([tweet_text]).toarray()
			# Load your Logistic_regression.pkl file 
            predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            results = model_app.classify_desc(format(prediction))
			# When model has successfully run, will print prediction
            st.success("Your tweet is classified as: {} ".format(results))

    if model == 'K-Nearest Neighbour' :
        st.success("K-Nearest Neighbour is a machine learning model that works by storing training data and only making predictions once it is called upon. It essentially takes these training data points and uses them as references in order to plot and classify the test data. It does this by looking at the similarity of data points then uses that as a guide to classify a new point." )
        tweet_text = st.text_area("Type a tweet")
        tweet_text = model_app.cleaning_text(tweet_text)
        if st.button("Classify"):
			# Transforming user input with vectorizer
            vect_text = tweet_vect.transform([tweet_text]).toarray()
			# Load your Logistic_regression.pkl file 
            predictor = joblib.load(open(os.path.join("resources/K_Neighbors.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            results = model_app.classify_desc(format(prediction))
			# When model has successfully run, will print prediction
            st.success("Your tweet is classified as: {} ".format(results))

    if model == 'Naive_Bayes' :
        st.success("A Naive Bayes model is a model that is usually used for classifying discrete variables.What makes this model unique however, is that it can also be used for text classification problems. Instead of just counting the presence of a word in a document, this model looks at the frequency of each word in a document. This helps the model in being able to generalize the data over a number of iterations. In addition it is a quick model to train because it only focuses on the probability of each class given a certain input variable. It does not take into account other features such as coeffients." )
        tweet_text = st.text_area("Type a tweet")
        tweet_text = model_app.cleaning_text(tweet_text)
        if st.button("Classify"):
			# Transforming user input with vectorizer
            vect_text = tweet_vect.transform([tweet_text]).toarray()
			# Load your Logistic_regression.pkl file 
            predictor = joblib.load(open(os.path.join("resources/Naive_Bayes.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            results = model_app.classify_desc(format(prediction))
			# When model has successfully run, will print prediction
            st.success("Your tweet is classified as: {} ".format(results))

    if model == 'Linear_Support_Vector' :
        st.success("A linear support vector classifier is a model that works by linearly separating data, into two classes.It works by the assumption that when classifying distinct classes can be created without any crossing over of classes." )
        tweet_text = st.text_area("Type a tweet")
        tweet_text = model_app.cleaning_text(tweet_text)
        if st.button("Classify"):
			# Transforming user input with vectorizer
            vect_text = tweet_vect.transform([tweet_text]).toarray()
			# Load your Logistic_regression.pkl file 
            predictor = joblib.load(open(os.path.join("resources/Linear_Support_Vector.pkl"),"rb"))
            prediction = predictor.predict(vect_text)
            results = model_app.classify_desc(format(prediction))
			# When model has successfully run, will print prediction
            st.success("Your tweet is classified as: {} ".format(results))
elif selected == 'Improvements':
    st.subheader("Comments")
    st.info("In the near future we will look into implementing a feature that will enable the user display visuals foreach model in real time")
    #st_lottie(load_lottie_conclusion, speed=1, loop=True, width=350,  quality="high", reverse=True)



        
