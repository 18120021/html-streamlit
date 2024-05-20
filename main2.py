import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import tempfile
from knn import knn_classify, knn_score
from svm import svm_classify, svm_score
from lgbm import lgbm_classify, lgbm_score
from stacking1 import st1_classify, st1_score
from stacking2 import st2_classify, st2_score
from stacking3 import st3_classify, st3_score
warnings.filterwarnings("ignore")  # Ignore warnings

# load Style css
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main dashboard
st.markdown(
    '<style>body { background-color: #ffc0cb; }</style>'
    '<h1 class="title" style="text-align: center;">Traffic Classification Data Dashboard</h1>'
    '</div>', unsafe_allow_html=True)

# Sidebar - Select machine learning model
model_ml = st.sidebar.selectbox("Select Machine Learning Model", ["Individual", "Stacking"], index=None)

if model_ml == "Individual":
    # Sidebar - Select machine learning model
    selected_method = st.sidebar.selectbox("Select Machine Learning Method", ["KNN", "SVM", "LightGBM"], index=None)
    # Load data based on the selected model
    if selected_method == "KNN":
        classify_function = knn_classify.classify
        score_function = knn_score.score
    elif selected_method == "SVM":
        classify_function = svm_classify.classify
        score_function = svm_score.score
    elif selected_method == "LightGBM":
        classify_function = lgbm_classify.classify
        score_function = lgbm_score.score
    else:
        st.write("Please choose the Machine Learning Method")
        st.stop()
elif model_ml == "Stacking":
    # Sidebar - Select machine learning model
    selected_method = st.sidebar.selectbox("Select Machine Learning Model", ["Stacking (kNN, SVM)", "Stacking (kNN, LightGBM, SVM)", "Stacking (kNN, Random Forest, LightGBM, SVM)", ], index=None)
    # Load data based on the selected model
    if selected_method == "Stacking (kNN, SVM)":
        classify_function = st1_classify.classify
        score_function = st1_score.score
    elif selected_method == "Stacking (kNN, LightGBM, SVM)":
        classify_function = st2_classify.classify
        score_function = st2_score.score
    elif selected_method == "Stacking (kNN, Random Forest, LightGBM, SVM)":
        classify_function = st3_classify.classify
        score_function = st3_score.score
    else:
        st.write("Please choose the Machine Learning Method")
        st.stop()
else:
    st.write("Please choose the Machine Learning Model")
    st.stop()

# Load data
metrics_df = score_function()
traffic_types_df = classify_function()

# Display metrics values
st.subheader(f"Metrics Scoring {selected_method}")
col1, col2, col3, col4, col5 = st.columns(5, gap='small')
# Add a frame line between columns
st.markdown('<hr style="border-top: 1px solid #bbb;">', unsafe_allow_html=True)
for index, row in metrics_df.iterrows():
    if index == 0:
        with col1:
            st.info(row['Metric'])
            #convert to percentage with two decimal places
            st.write("<div style='text-align: center;'>{:.2f}%</div>".format(row['Value'] * 100), unsafe_allow_html=True)
    elif index == 1:
        with col2:
            st.info(row['Metric'])
            #convert to percentage with two decimal places
            st.write("<div style='text-align: center;'>{:.2f}%</div>".format(row['Value'] * 100), unsafe_allow_html=True)
    elif index == 2:
        with col3:
            st.info(row['Metric'])
            #convert to percentage with two decimal places
            st.write("<div style='text-align: center;'>{:.2f}%</div>".format(row['Value'] * 100), unsafe_allow_html=True)
    elif index == 3:
        with col4:
            st.info(row['Metric'])
            #convert to percentage with two decimal places
            st.write("<div style='text-align: center;'>{:.2f}%</div>".format(row['Value'] * 100), unsafe_allow_html=True)
    elif index == 4:
        with col5:
                st.info(row['Metric'])
                #convert to percentage with two decimal places
                st.write("<div style='text-align: center;'>{:.2f}%</div>".format(row['Value'] * 100), unsafe_allow_html=True)

# Convert 'date' column to datetime
traffic_types_df['date'] = pd.to_datetime(traffic_types_df['date']).dt.date

# Convert 'time' column to datetime format
traffic_types_df['time'] = pd.to_datetime(traffic_types_df['time'], format='%H:%M:%S').dt.time

# Filters
# Get unique predicted traffic types
selected_traffic_types = list(set(traffic_types_df["Predicted Traffic Type"].str.strip()))

# Multiselect dropdown with default options
predicted_traffic_types = st.sidebar.multiselect("Filter Predicted Traffic Type", options=selected_traffic_types, default=[])
date_option = st.sidebar.selectbox("Filter Date", options=["All Dates", "Custom"], index=None)

# Check if the user selected "Custom" option
if date_option == "Custom":
    start_date = pd.to_datetime(st.sidebar.date_input("Start Date"))
    end_date = pd.to_datetime(st.sidebar.date_input("End Date"))
elif date_option == "All Dates":
    start_date = traffic_types_df["date"].min()
    end_date = traffic_types_df["date"].max()

# Check if both filters are empty, if so, exit the script without displaying anything
if not predicted_traffic_types or not date_option:
    st.write("No data found for the selected filters. Please adjust your filters.")
    st.stop()

# Filtered dataframe
filtered_df = traffic_types_df.copy()
if predicted_traffic_types:
    filtered_df = filtered_df[filtered_df["Predicted Traffic Type"].isin(predicted_traffic_types)]
if date_option == "Custom":
    filtered_df = filtered_df[(filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)]

# Drop 'no' and 'traffic_type' columns
filtered_df = filtered_df.drop(columns=['no', 'traffic_type'])

# Display filtered data table sorted by date and time, and start with the 'date' column
st.subheader("Filtered Traffic Data Table")
st.write(filtered_df.sort_values(by=["date", "time"]).set_index('date'))

# Count the number of data for each predicted traffic type
predicted_traffic_counts = filtered_df["Predicted Traffic Type"].value_counts()

# Group the data by 'predicted_traffic_type' and 'date', and count the data for each group
grouped_data = filtered_df.groupby(['Predicted Traffic Type', 'date']).size().reset_index(name='count')

# Handle cases where only one Predicted Traffic Type, one Date, or both are selected
if len(predicted_traffic_types) == 1:
    st.subheader("Only one Predicted Traffic Type selected. Showing only table and line chart.")
    if len(filtered_df["date"].unique()) > 1:
        st.subheader("Traffic Type Over Time")
        fig_line = px.line(grouped_data, x="date", y="count", color="Predicted Traffic Type")
        fig_line.update_layout(width=800)  # Adjust width of the chart
        st.plotly_chart(fig_line, use_container_width=True, align="center")
elif date_option == "Custom" and start_date == end_date:
    st.subheader("Only one Date selected. Showing only table and pie chart.")
    if len(predicted_traffic_counts) > 1:
        st.subheader("Traffic Type Distribution")
        fig_pie = px.pie(values=predicted_traffic_counts, names=predicted_traffic_counts.index)
        fig_pie.update_layout(width=500)  # Adjust width of the chart
        st.plotly_chart(fig_pie, use_container_width=True, align="center")
elif len(predicted_traffic_types) == 1 and date_option == "Custom" and start_date == end_date:
    st.subheader("Only one Predicted Traffic Type and one Date selected. Showing only table.")
else:
    col1, col2 = st.columns((2))
    # Show both chart
    with col1:
        if len(predicted_traffic_counts) > 1:
            st.subheader("Traffic Type Distribution")
            fig_pie = px.pie(values=predicted_traffic_counts, names=predicted_traffic_counts.index)
            fig_pie.update_layout(width=350, height=400)  # Adjust width of the chart
            st.plotly_chart(fig_pie)
    # Show line chart
    with col2:
        if len(filtered_df["date"].unique()) > 1:
            st.subheader("Traffic Type Over Time")
            fig_line = px.line(grouped_data, x="date", y="count", color="Predicted Traffic Type")
            fig_line.update_layout(width=480, height=400)  # Adjust width of the chart
            st.plotly_chart(fig_line)

# Provide link to download classified data
# Create a temporary file to store the data
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    traffic_types_df.to_csv(tmp.name, index=False)
    tmp.close()
    with open(tmp.name, 'rb') as f:
        data = f.read()

# Offer the temporary file for download
st.download_button(label="Download Classified New File", data=data, file_name='traffic_type_prediction.csv', mime='text/csv')