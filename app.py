import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import pickle, joblib
import pymysql
import base64
import time

# Load the saved model and preprocessor
model = pickle.load(open('rfc.pkl', 'rb'))
preprocessor = joblib.load('preprocessor.sav')  # Load the preprocessor

# Function to encode local image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Set background image using a local file
def set_background_image(image_path):
    base64_img = get_base64_image(image_path)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background_image('new image.jpg')  # Update this line with the correct file name

# Add animated title
def add_title_animation():
    title_html = '''
    <style>
    @keyframes glow {
        0% { text-shadow: 0 0 5px #FF5733, 0 0 10px #FF5733; }
        50% { text-shadow: 0 0 20px #FFBD33, 0 0 30px #FFBD33; }
        100% { text-shadow: 0 0 5px #FF5733, 0 0 10px #FF5733; }
    }
    .title-text {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        color: #FFBD33;
        animation: glow 2s infinite alternate;
    }
    </style>
    <div class="title-text">Downtime Prediction</div>
    '''
    st.markdown(title_html, unsafe_allow_html=True)

# Add animated welcome text
def add_text_animation():
    animated_text = '''
    <style>
    @keyframes slide-fade-in {
        0% { opacity: 0; transform: translateX(-100%); }
        100% { opacity: 1; transform: translateX(0); }
    }
    .welcome-text {
        font-size: 2.5em;
        font-weight: bold;
        color: #FFBD33;
        animation: slide-fade-in 2s ease-in-out;
        text-shadow: 2px 2px 4px #000000;
    }
    </style>
    <div class="welcome-text">Welcome to the <span style="color:#FF5733;">Downtime Prediction</span> App!</div>
    '''
    st.markdown(animated_text, unsafe_allow_html=True)

# Add a loading spinner
def show_loading_spinner():
    with st.spinner("Predicting downtime... Please wait ⏳"):
        time.sleep(2)  # Simulate a delay for prediction

# Preprocess data
def preprocess_data(data):
    data = data.drop(['Assembly_Line_No', 'Date', 'Machine_ID'], axis=1, errors='ignore')
    processed_data = preprocessor.transform(data)
    return processed_data

# Predict downtime
def predict(data, user=None, pw=None, db=None):
    try:
        processed_data = preprocess_data(data)
        show_loading_spinner()
        predictions = pd.DataFrame(model.predict(processed_data), columns=['Downtime'])
        final = pd.concat([predictions, data.drop('Downtime', axis=1, errors='ignore')], axis=1)
        if user and pw and db:
            engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
            final.to_sql('Mc_DT', con=engine, if_exists='replace', chunksize=200, index=False)
            return final, "Results have been saved to the database."
        else:
            return final, "Results are ready. Database credentials were not provided."
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame(), "Error occurred during processing."

# Main function
def main():
    add_title_animation()
    add_text_animation()
    st.sidebar.title("File Upload and Database Credentials")

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False)
    user = st.sidebar.text_input("Username", "")
    pw = st.sidebar.text_input("Password", type='password', value="")
    db = st.sidebar.text_input("Database Name", "")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("Data preview:")
            st.dataframe(data.head())
            if st.button("Predict Downtime 🚀"):
                result, message = predict(data, user, pw, db)
                if not result.empty:
                    st.success(message)
                    st.write("Predictions:")
                    st.dataframe(result)
                else:
                    st.warning(message)
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.sidebar.info("Upload a CSV or Excel file to get started.")

if __name__ == '__main__':
    main()
