import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore ", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Global superstore Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

df = pd.read_csv("GlobalSuperstoreliteOriginal.csv", encoding="ISO-8859-1")

col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Getting the min and max date
startDate = pd.to_datetime(df["Order Date"]).min()
endDate = pd.to_datetime(df["Order Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

st.sidebar.header("Pick to view analysis: ")

# Filters for Category and Sub-Category
category = st.sidebar.multiselect("Pick your Category", df["Category"].unique())
if not category:
    df2 = df.copy()
else:
    df2 = df[df["Category"].isin(category)]

subcategory = st.sidebar.multiselect("Pick your Sub-Category", df2["Sub-Category"].unique())
if not subcategory:
    df3 = df2.copy()
else:
    df3 = df2[df2["Sub-Category"].isin(subcategory)]

filtered_df = df3.copy()

# Time Series Analysis
time_series_data = filtered_df.groupby(filtered_df["Order Date"].dt.to_period("M")).agg({"Sales": "sum", "Profit": "sum"}).reset_index()
time_series_data["Order Date"] = time_series_data["Order Date"].dt.strftime("%Y-%m")

# Plot Time Series
st.subheader('Sales and Profit over time')
fig_time_series = px.line(time_series_data, x="Order Date", y=["Sales", "Profit"], 
                          labels={"value": "Amount", "variable": "Metric"},
                          title='Sales and Profit Trend Over Time', height=500, width=1000, template="gridon")
st.plotly_chart(fig_time_series, use_container_width=True)

# Bar Chart and Pie Chart in Same Row
st.subheader('Quantity by Category and Segment-wise Sales')
with st.expander("Click to show/hide"):
    fig_bar_pie = px.bar(df2.groupby("Category").size().reset_index(name="Quantity"), x="Category", y="Quantity", 
                         labels={"Quantity": "Number of Items Sold"}, 
                         title='Quantity by Category', template="seaborn")
    fig_pie = px.pie(df2, names='Segment', title='Segment-wise Sales')
    st.plotly_chart(fig_bar_pie, use_container_width=True)
    st.plotly_chart(fig_pie, use_container_width=True)

# Scatter Plot
st.subheader('Scatter Plot for Profit and Discount')
fig_scatter = px.scatter(filtered_df, x="Profit", y="Discount", color="Category",
                         labels={"Profit": "Profit", "Discount": "Discount"},
                         title='Scatter Plot for Profit and Discount by Category', template="seaborn")
st.plotly_chart(fig_scatter, use_container_width=True)

# Association Rules and Heatmap
st.subheader('Association Rules and Heatmap')

# Filter for min support
min_support = st.sidebar.slider("Minimum Support", min_value=0.01, max_value=0.5, value=0.05, step=0.01)

# Filter for min threshold
min_threshold = st.sidebar.slider("Minimum Threshold", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

association_df = filtered_df[['Category', 'Sub-Category']]
one_hot_encoded = pd.get_dummies(association_df)
frequent_itemsets = apriori(one_hot_encoded, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)

# Convert frozensets to strings
rules['antecedents'] = rules['antecedents'].apply(lambda x: ", ".join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ", ".join(list(x)))

# Create a heatmap of the association rules using Matplotlib
heatmap_data = pd.crosstab(rules['antecedents'], rules['consequents'])
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, cbar=False, ax=ax)
st.pyplot(fig)

# View Button for Summary
if st.button("View Summary"):
    st.subheader("Summary Table")
    st.write(filtered_df.groupby(['Category', 'Sub-Category']).agg({'Sales': 'sum', 'Quantity': 'sum'}).reset_index())

# Download Button for Summary
if st.button("Download Summary"):
    csv = filtered_df.groupby(['Category', 'Sub-Category']).agg({'Sales': 'sum', 'Quantity': 'sum'}).reset_index().to_csv(index=False)
    st.download_button(label="Download Summary (CSV)", data=csv, file_name="summary.csv")