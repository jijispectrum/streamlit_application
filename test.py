# import streamlit as st
# import pickle
# import re

# # Load the pre-trained machine learning model
# model_file_path = "trained_model(2).pkl"
# with open(model_file_path, "rb") as f:
#     model = pickle.load(f)

# # Function to preprocess text
# def preprocess_text(text):
#     text = text.lower()  # Convert text to lowercase
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     return text

# # Function to predict sentiment
# def predict_sentiment(text):
#     processed_text = preprocess_text(text)
#     prediction = model.predict([processed_text])[0]
#     return "Positive" if prediction == 1 else "Negative"

# # Streamlit app
# def main():
#     st.title("Sentiment Prediction App")

#     # Text input for user to enter new text
#     user_input = st.text_input("Enter text to analyze sentiment:")

#     # Button to perform sentiment analysis
#     if st.button("Predict Sentiment"):
#         if user_input:
#             # Predict sentiment using the pre-trained model
#             sentiment = predict_sentiment(user_input)

#             # Display sentiment prediction
#             st.write(f"Sentiment: {sentiment}")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import joblib
# import re

# # Load the pre-trained machine learning model
# model_file_path = "trained_model(2).joblib"
# model = joblib.load(model_file_path)

# # Function to preprocess text
# def preprocess_text(text):
#     text = text.lower()  # Convert text to lowercase
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     return text

# # Function to predict sentiment
# def predict_sentiment(text):
#     processed_text = preprocess_text(text)
#     prediction = model.predict([processed_text])[0]
#     return "Positive" if prediction == 1 else "Negative"

# # Streamlit app
# def main():
#     st.title("Sentiment Prediction App")

#     # Text input for user to enter new text
#     user_input = st.text_input("Enter text to analyze sentiment:")

#     # Button to perform sentiment analysis
#     if st.button("Predict Sentiment"):
#         if user_input:
#             # Predict sentiment using the pre-trained model
#             sentiment = predict_sentiment(user_input)

#             # Display sentiment prediction
#             st.write(f"Sentiment: {sentiment}")

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess the data
df = pd.read_csv("newtwitter(2).csv", encoding='latin1')  # Replace "your_data.csv" with the path to your CSV file
# Perform any necessary preprocessing steps here
df.dropna(subset=['text'], inplace=True)
# Handle missing values in text data
df['text'].fillna('', inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a sentiment analysis model
logistic_regression_classifier = LogisticRegression(max_iter=1000)
logistic_regression_classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = logistic_regression_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy}")

# Save the model
joblib.dump(logistic_regression_classifier, "logistic_regression_classifier.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # Text input for user to enter new text
    user_input = st.text_input("Enter text to analyze sentiment:")

    # Button to perform sentiment analysis
    if st.button("Analyze Sentiment"):
        if user_input:
            # Vectorize the user input
            user_input_tfidf = tfidf_vectorizer.transform([user_input])

            # Predict sentiment using the trained model
            prediction = logistic_regression_classifier.predict(user_input_tfidf)[0]

            # Display the sentiment prediction
            st.write(f"Predicted Sentiment: {prediction}")

            # Display an image
            if prediction == 'positive':
                st.image('positive.jpeg', caption='Positive Image', width=300)
            elif prediction == 'negative':
                st.image('Negtative.jpeg', caption='Negative Image', width=300)
            else:
                st.image('Neutral.jpeg', caption='Neutral Image', width=300)

if __name__ == "__main__":
    main()
