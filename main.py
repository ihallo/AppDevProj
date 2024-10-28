import streamlit as st
from io import StringIO 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Downloading Dataset from Kaggle
import kagglehub
import os

# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation

# Download latest version
path = kagglehub.dataset_download("uom190346a/ai-powered-job-market-insights")

print("Path to dataset files:", path)

#List the files in the downloaded dataset directory
files = os.listdir(path)
print("Files in the downloaded dataset directory:", files)

# Load the dataset (make sure to replace 'weather_data.csv' with the actual file name)
file_name = 'ai_job_market_insights.csv'  # Adjust as necessary
file_path = os.path.join(path, file_name)

# Read the CSV file
df = pd.read_csv(file_path)

st.title("Job Market Insights")

st.subheader("Dataset Preview")
st.write(df.head())

# Column Descriptions
st.subheader("Column Descriptions")

column_descriptions = {
    "Job_Title": "The title of the job position.",
    "Industry": "The type of industry that is hiring.",
    "Company_Size": "The company size of the job.",
    "Location": "The location of the job.",
    "AI_Adoption_Level": "The level to which the company has integrated AI into its operations.",
    "Automation_Risk": "The probability that the job will be automated during the next years.",
    "Required_Skills": "The primary skills required for the job role.",
    "Salary_USD": "The job offers an annual pay in USD.",
    "Remote_Friendly": "Determines whether the job may be completed remotely.",
    "Job_Growth_Projection": "The expected growth or drop in the job position over the following five years."
}

for column, description in column_descriptions.items():
    st.markdown(f"**{column}:** {description}")       
    
st.subheader("DataFrame Information")
buffer = StringIO()  
df.info(buf=buffer)  
info = buffer.getvalue()  
st.text(info)

st.write("### **Observation**")
st.write("As we can see from the DataFrame information, there are no null values in any of the data types. Based on this, we can conclude that there are no data columns that need to be dropped.")

st.subheader("Duplicate Rows")
st.write(f"Number of duplicate rows: {df.duplicated().sum()}")  

duplicates = df[df.duplicated(keep=False)]
st.dataframe(duplicates)

st.write("### **Observation**")
st.write("There are no duplicate rows found on the dataframe.")

# Categorical columns
cat_col = [col for col in df.columns if df[col].dtype == 'object']
st.write('#Categorical columns:', cat_col)

# Numerical columns
num_col = [col for col in df.columns if df[col].dtype != 'object']
st.write('Numerical columns :',num_col)


cat_unique_counts = df[cat_col].nunique().reset_index()
cat_unique_counts.columns = ['Column', 'Unique Counts']
st.subheader('Unique Value Counts for Categorical Columns')
st.dataframe(cat_unique_counts)

num_unique_counts = df[num_col].nunique().reset_index()
num_unique_counts.columns = ['Column', 'Unique Counts']
st.subheader('Unique Value Counts for Numerical Columns')
st.dataframe(num_unique_counts)


for col in cat_col:
    st.write(f'Unique values in {col}:', df[col].unique())
    
for col in num_col:
    st.write(f'Unique values in {col}:', df[col].unique())
    
st.write("### **Observation**")
st.write("We used df[cat_col].nunique() to identify how many unique data in each categorized columns. Since there is only one numerical which is Salary_USD and we identified that all categorical columns are properly intialized with the use of .unique().")

df_data = pd.DataFrame()

job_encoder = LabelEncoder()
industry_encoder = LabelEncoder()
size_encoder = LabelEncoder()
location_encoder = LabelEncoder()
ai_adoption_encoder = LabelEncoder()
automation_encoder = LabelEncoder()
skills_encoder = LabelEncoder()
remote_encoder = LabelEncoder()
growth_encoder = LabelEncoder()

df_data['Job_encoded'] = job_encoder.fit_transform(df['Job_Title'])
df_data['Industry_encoded'] = industry_encoder.fit_transform(df['Industry'])
df_data['Size_encoded'] = size_encoder.fit_transform(df['Company_Size'])
df_data['Location_encoded'] = location_encoder.fit_transform(df['Location'])
df_data['AI_Adoption_encoded'] = ai_adoption_encoder.fit_transform(df['AI_Adoption_Level'])
df_data['Automation_encoded'] = automation_encoder.fit_transform(df['Automation_Risk'])
df_data['Skills_encoded'] = skills_encoder.fit_transform(df['Required_Skills'])
df_data['Remote_encoded'] = remote_encoder.fit_transform(df['Remote_Friendly'])
df_data['Growth_encoded'] = growth_encoder.fit_transform(df['Job_Growth_Projection'])
df_data['Salary_USD'] = df['Salary_USD']

st.subheader('Encoding Process:')
st.code("""
job_encoder = LabelEncoder() 
industry_encoder = LabelEncoder()
size_encoder = LabelEncoder()
location_encoder = LabelEncoder()
ai_adoption_encoder = LabelEncoder()
automation_encoder = LabelEncoder()
skills_encoder = LabelEncoder()
remote_encoder = LabelEncoder()
growth_encoder = LabelEncoder()

df_data['Job_encoded'] = job_encoder.fit_transform(df['Job_Title'])
df_data['Industry_encoded'] = industry_encoder.fit_transform(df['Industry'])
df_data['Size_encoded'] = size_encoder.fit_transform(df['Company_Size'])
df_data['Location_encoded'] = location_encoder.fit_transform(df['Location'])
df_data['AI_Adoption_encoded'] = ai_adoption_encoder.fit_transform(df['AI_Adoption_Level'])
df_data['Automation_encoded'] = automation_encoder.fit_transform(df['Automation_Risk'])
df_data['Skills_encoded'] = skills_encoder.fit_transform(df['Required_Skills'])
df_data['Remote_encoded'] = remote_encoder.fit_transform(df['Remote_Friendly'])
df_data['Growth_encoded'] = growth_encoder.fit_transform(df['Job_Growth_Projection'])
df_data['Salary_USD'] = df['Salary_USD']
""")

st.write("### **Observation**")
st.write(" We encoded all object datatypes for RandomForestClassifier.", "We also created a new DataFrame to store the encoded data.")

features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Growth_encoded','Salary_USD']
X1 = df_data[features]
Y1 = df_data['Automation_encoded']

st.subheader('Features (X1):')
st.write(X1)

st.subheader('Target Variable (Y1):')
st.write(Y1)

st.write("### **Observation**")
st.write(" To prepare for machine learning we categorized the needed datatype for predicition inside X. While we put the data will be predicted to Y which is Automation_Risk.")

# Min-Max Scaling (Normalization)

scaler = MinMaxScaler(feature_range=(0, 1))

num_col_ = [col for col in X1.columns if X1[col].dtype != 'object']
X1 = X1

X1[num_col_] = scaler.fit_transform(X1[num_col_])
X1.head()

st.subheader('Min-Max Scaling (Normalization)')

scaler = MinMaxScaler(feature_range=(0, 1))

num_col_ = [col for col in X1.columns if X1[col].dtype != 'object']
X1[num_col_] = scaler.fit_transform(X1[num_col_])


st.write('Scaled Features (X1):')
st.dataframe(X1)

st.write("### **Observation**")
st.write(" We used Min-Max scaling for standardization to help the machine learning. Since, standardization makes the data more suitable for algorithms that assume a Gaussian distributuion or require features to have zero mean and unit variance. As we can see now, the Salary_USD was converted to a decimal from a whole number.")

st.subheader('Automation Risk Distribution')

automationRisk_list = df['Automation_Risk'].unique().tolist()
automationRisk_counts = df['Automation_Risk'].value_counts()
automationRisk_counts_list = automationRisk_counts.tolist()

def pie_chart_AutomationRisk():

 
  plt.pie(automationRisk_counts_list, labels=automationRisk_list, autopct='%1.1f%%')
  plt.title('Pie Chart of Automation Risk')
  st.pyplot(plt)
pie_chart_AutomationRisk()

st.write("### **Observation**")
st.write("We created a piechart for our Automation Risk (our first Y) and we can see that it is almost evenly distributed. Which is good for our machine learning to not have any biases.")

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=42)

st.markdown("#### **X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=42)**")

st.write('Training Features (X1_train):')
st.dataframe(X1_train)

st.write('Testing Features (X1_test):')
st.dataframe(X1_test)

st.write('Training Target Variable (Y1_train):')
st.dataframe(Y1_train)

st.write('Testing Target Variable (Y1_test):')
st.dataframe(Y1_test)

st.write("### **Observation**")
st.markdown("""
*  We used `.head()` to give us a preview of the `X1_test` and `X1_train` data.
*  We used `.shape` to give us the dimensions of the `X1_test` and `X1_train` data.

 As you can see, the test shape has a lower number than the training shape. Since we should always allot more data to training for the machine learning model to practice.
""")

st.write('Training Features (Y1_train):')
st.dataframe(Y1_train)

st.write('Testing Features (Y1_test):')
st.dataframe(Y1_test)

st.write('Training Target Variable (Y1_train):')
st.dataframe(Y1_train)

st.write('Testing Target Variable (Y1_test):')
st.dataframe(Y1_test)

clf = RandomForestClassifier(random_state=42)
clf.fit(X1_train, Y1_train)