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

# Set your local image path here
set_background_image('download (4)')

# Add animated text
def add_text_animation():
    animated_text = '''
    <style>
    @keyframes slide-fade-in {{
        0% {{
            opacity: 0;
            transform: translateX(-100%);
        }}
        100% {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    .animated-text {{
        font-size: 2.5em;
        font-weight: bold;
        color: #FFFFFF;
        animation: slide-fade-in 2s ease-in-out;
        text-shadow: 2px 2px 4px #000000;
    }}
    </style>

    <div class="animated-text">
        Welcome to Downtime Prediction App!
    </div>
    '''
    st.markdown(animated_text, unsafe_allow_html=True)

# Add a spinning gear animation
def add_gear_animation():
    gear_animation = '''
    <style>
    @keyframes spin {{
        0% {{
            transform: rotate(0deg);
        }}
        100% {{
            transform: rotate(360deg);
        }}
    }}
    .gear {{
        display: inline-block;
        font-size: 3em;
        animation: spin 2s linear infinite;
        color: #FF5733;
    }}
    </style>

    <div class="gear">
        ⚙️
    </div>
    '''
    st.markdown(gear_animation, unsafe_allow_html=True)

# Add a loading spinner
def show_loading_spinner():
    with st.spinner("Predicting downtime... Please wait ⏳"):
        time.sleep(2)  # Simulate a delay for prediction

# Preprocess data
def preprocess_data(data):
    """
    Preprocess the input data using the saved preprocessor.
    """
    # Drop unnecessary columns
    data = data.drop(['Assembly_Line_No', 'Date', 'Machine_ID'], axis=1, errors='ignore')
    
    # Apply the preprocessor
    processed_data = preprocessor.transform(data)
    return processed_data

# Predict downtime
def predict(data, user=None, pw=None, db=None):
    """
    Predict downtime based on the input data and optionally save results to a MySQL database.
    """
    try:
        # Preprocess the data
        processed_data = preprocess_data(data)

        # Show loading spinner
        show_loading_spinner()

        # Predict using the model
        predictions = pd.DataFrame(model.predict(processed_data), columns=['Downtime'])

        # Combine predictions with the original data
        final = pd.concat([predictions, data.drop('Downtime', axis=1, errors='ignore')], axis=1)

        if user and pw and db:
            # Create database connection
            engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

            # Save to database
            final.to_sql('Mc_DT', con=engine, if_exists='replace', chunksize=200, index=False)
            return final, "Results have been saved to the database."
        else:
            return final, "Results are ready. Database credentials were not provided."

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame(), "Error occurred during processing."

# Main function
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Downtime Prediction 🏭")
    st.sidebar.title("File Upload and Database Credentials")

    # Add animated text
    add_text_animation()

    # Add gear animation
    add_gear_animation()

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx'], accept_multiple_files=False)

    # Database credentials input
    user = st.sidebar.text_input("Username", "")
    pw = st.sidebar.text_input("Password", type='password', value="")
    db = st.sidebar.text_input("Database Name", "")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

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
