import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(
    page_title="Call Service Predictor",
    page_icon="📞"
)

df = pd.read_csv('Call_Center_Data.csv')
df['Answer Rate'] = df['Answer Rate'].str.replace('%', '').astype(float)

def hms_to_seconds(hms):
    h, m, s = map(int, hms.split(':'))
    return h * 3600 + m * 60 + s

df['Answer Speed_AVG'] = df['Answer Speed_AVG'].apply(hms_to_seconds)
df['Talk Duration_AVG'] = df['Talk Duration_AVG'].apply(hms_to_seconds)
df['Waiting Time_AVG'] = df['Waiting Time_AVG'].apply(hms_to_seconds)

df['Service Level'] = df['Service Level'].str.replace('%', '').str.replace(r'[^0-9.]+', '', regex=True).str.strip().astype(float)

x = df[["Answer Rate", "Answer Speed_AVG", "Talk Duration_AVG", "Waiting Time_AVG"]]
y = df["Service Level"]

model = LinearRegression()
model.fit(x, y)

st.markdown("""
    <style>
        body {
            background-color: var(--background-color, #ffffff);
            color: var(--text-color, #000000);
        }
        .stApp {
            background-color: var(--background-color, #ffffff);
        }
        .stMarkdown, .stTextInput, .stNumberInput, .stButton {
            color: var(--text-color, #000000);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --background-color: #121212;
                --text-color: #ffffff;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.title("Service Level Prediction for Call Center")
st.subheader("Enter Details:")

Answer_Rate = st.number_input("Answer Rate (%)", min_value=0, max_value=100, step=1)
Answer_Speed_AVG = st.text_input("Answer Speed (AVG) in HH:MM:SS", "0:00:00")
Talk_Duration_AVG = st.text_input("Talk Duration (AVG) in HH:MM:SS", "0:00:00")
Waiting_Time_AVG = st.text_input("Waiting Time (AVG) in HH:MM:SS", "0:00:00")

def hms_to_seconds_input(hms):
    try:
        h, m, s = map(int, hms.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        st.error("Invalid time format! Please use HH:MM:SS.")
        return None

predict = st.button("Predict Service Level")

if predict:
    Answer_Speed_AVG_sec = hms_to_seconds_input(Answer_Speed_AVG)
    Talk_Duration_AVG_sec = hms_to_seconds_input(Talk_Duration_AVG)
    Waiting_Time_AVG_sec = hms_to_seconds_input(Waiting_Time_AVG)

    if None not in (Answer_Speed_AVG_sec, Talk_Duration_AVG_sec, Waiting_Time_AVG_sec):
        prediction = model.predict([[Answer_Rate, Answer_Speed_AVG_sec, Talk_Duration_AVG_sec, Waiting_Time_AVG_sec]])
        st.success(f"Predicted Service Level: {prediction[0]:.2f}%")
    else:
        st.error("Please correct the input errors before predicting.")
