

import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://anhdepfree.com/wp-content/uploads/2019/05/50-anh-background-dep-nhat-4.jpg');
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);

}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.pexels.com/photos/2088203/pexels-photo-2088203.jpeg?auto=compress&cs=tinysrgb&w=600');
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


	st.info("This application is all about tweet sentiment analysis of climate change. It is able to classify whether" 
			 "or not a person believes in climate change, based on their novel tweet data.")


	# Building out the "Tweet Sentitment analysis " page		
	if selection == "Tweet analysis":
		st.info("This app analyses sentiments on climate change based on tweet data")
		#top level filters
		#message_filter = st.selectbox("Select the message", pd.unique(raw['sentiment']))
		# dataframe filter
		#df = raw[raw['sentiment']== message_filter] 
		st.markdown("### Tweet distribution")
		sentiment = raw['sentiment'].value_counts()
		sentiment = pd.DataFrame({'Sentiment':sentiment.index, 'Tweets':sentiment.values})
	
		# create two columns for charts
		fig_col1, fig_col2 = st.columns(2)
		
		with fig_col1:
			fig = fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
			st.plotly_chart(fig)
	
       	#
		with fig_col2:
			fig = px.pie(sentiment, values= 'Tweets', names= 'Sentiment')
			st.plotly_chart(fig)
			
		


"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies

import streamlit as st
import joblib,os
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
rfc_vectorizer = open("resources/rfc_TfidfVectorizer.pkl","rb")
tweet_rfc = joblib.load(rfc_vectorizer)



# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way

	options = ["Background","Know your file","Prediction"]
	selection = st.sidebar.selectbox("Lets interact", options)

	# Building out the "Information" page
	if selection == "Background":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")
		image = Image.open(os.path.join("resources/imgs/twitter_logo.jpg"))
		st.image(image, caption='Sunrise by the mountains')

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		model = st.radio(
    	"Select a model to classifiy your tweet",
    		('Random Forest Classifier', 'Logistic_regression'))
		# Creating a text box for user input
		# upload a file
		data = st.radio(
    	"How do you want to load data",
    		('Upload tweets samples', 'Type your tweet'))

		if data == 'Upload tweets samples' :
			upload_file = st.file_uploader("Upload file")
		else:
			tweet_text = st.text_area("Type a tweet")

		if model == 'Random Forest Classifier' :
			if st.button("Classify"):
				# Transforming user input with vectorizer
				if data == 'Upload tweets samples' :
					rfc_file = tweet_rfc.transform([upload_file]).toarray()
				else:
					rfc_text = tweet_rfc.transform([tweet_text]).toarray()
				
				# Load your randomfc_model.pkl file 
				predictor = joblib.load(open(os.path.join("resources/randomfc_model.pkl"),"rb"))
				if data == 'Upload tweets samples' :
					prediction_file  = predictor.predict(rfc_file)
				else:
					prediction = predictor.predict(rfc_text)
				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(prediction))
	
		if model == 'Logistic_regression' :
			if st.button("Classify"):
				# Transforming user input with vectorizer
				if data == 'Upload tweets samples' :
					vect_file = tweet_cv.transform([upload_file]).toarray()
				else:
					vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your Logistic_regression.pkl file 
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# When model has successfully run, will print prediction
				st.success("Text Categorized as: {}".format(prediction))

	#Building out the predication page
	if selection == "Random Forest Classifier":
		st.info("Just a little bit about the random classifyer model")
		
		image = Image.open(os.path.join("resources/imgs/twitter_logo.jpg"))
		st.image(image, caption='Sunrise by the mountains')
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			predictor = joblib.load(open(os.path.join("resources/randomfc_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

