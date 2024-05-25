import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Function to load and preprocess the dataset
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = pd.to_datetime(df['time'], format='%I:%M:%S %p').dt.time
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    df['duration'] = df.groupby(['date', 'position'])['datetime'].diff().dt.total_seconds().fillna(0)
    df['position'] = df['position'].str.lower()  # Normalize position values
    return df


# Streamlit application
st.title("Activity Data Analysis")

uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.write("Dataset Preview:")
    st.write(df.head())

    # Datewise total duration for each inside and outside
    duration_data = df.groupby(['date', 'position'])['duration'].sum().unstack(fill_value=0)

    st.write("Datewise Total Duration (Inside vs Outside):")
    st.line_chart(duration_data)

    # Datewise number of picking and placing activities
    activity_data = df.groupby(['date', 'activity']).size().unstack(fill_value=0)

    st.write("Datewise Number of Picking and Placing Activities:")
    st.bar_chart(activity_data)

    # Show the processed data for verification
    st.write("Processed Data:")
    st.write(duration_data)
    st.write(activity_data)

    # Matplotlib visualizations for better customization (optional)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    duration_data.plot(ax=ax[0], kind='line')
    ax[0].set_title('Datewise Total Duration (Inside vs Outside)')
    ax[0].set_ylabel('Total Duration (seconds)')

    activity_data.plot(ax=ax[1], kind='bar')
    ax[1].set_title('Datewise Number of Picking and Placing Activities')
    ax[1].set_ylabel('Number of Activities')

    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")
