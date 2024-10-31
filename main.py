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

st.write("# Automation Risk Prediction")

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

st.code("""X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=42)""")

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
* Same case with the Y1_train and Y1_test.

 As you can see, the test shape has a lower number than the training shape. Since we should always allot more data to training for the machine learning model to practice.
""")

clf = RandomForestClassifier(random_state=42)
clf.fit(X1_train, Y1_train)

st.code("""
* clf = RandomForestClassifier(random_state=42)
* clf.fit(X1_train, Y1_train)""")

st.write("RandomForestClassifier model trained successfully!")

st.write("### **Observation**")
st.write(" We used RandomForest Classifier algorithm for our supervised learning. To improve prediction accuracy and reduce overfitting.")

st.markdown("#### **Model Evaluation**")

Y1_pred = clf.predict(X1_test) #Prediction

st.code("""Y1_pred = clf.predict(X1_test) #Prediction""")

# Performance evaluation
accuracy = accuracy_score(Y1_test, Y1_pred)
classification_rep = classification_report(Y1_test, Y1_pred, target_names=automation_encoder.classes_)

st.write("### Classification Report")
st.text(classification_rep)

accuracy = accuracy_score(Y1_test, Y1_pred)
st.write("### Accuracy:")
st.markdown(f'{accuracy * 100:.2f}%')

st.write("### **Observation**")
st.write(" As we predict the Y and Based on the Performance evaluation - We can see the accuracy is low.")

feature_importance = clf.feature_importances_
st.code("""feature_importance = clf.feature_importances_""")
st.write("### feature_importance:")
st.write(feature_importance)

importance_df = pd.DataFrame({
    'Feature': X1.columns,
    'Importance': feature_importance
})

# Sort the DataFrame by importance for better readability
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
st.code("""importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)""")
# Display the resulting DataFrame
st.write("### importance_df:")
st.write(importance_df)

st.write("### **Observation**")
st.markdown("""
*  With the use of feature_importance we can identify each columns weighs in the model's capability of making decisions.
*  We can see from the results that Salary, Industry, Job Title, Location, and Skills weighs significantly more than the rest of the columns as we trained.
""")

fig = plt.figure(figsize=(20, 10))

st.write("### Tree plot:")
st.code("""
       * plt.figure(figsize=(20, 10))
       * tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=automationRisk_list, rounded=True, proportion=True)
""")

tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=automationRisk_list, rounded=True, proportion=True)
st.pyplot(fig)

st.write("### **Observation**")
st.write("The tree will help us visualize how the RFC decides.")

st.subheader("Training a better model.")
st.write("Our goal now is to balance out all of the classes.")

new_df = df
st.write("### New dataframe:")
st.code("""new_df = df""")
st.write(new_df.head())

st.write("### **Observation**")
st.write("We transfered df to  new_df.")


st.write("### automationRisk_counts:")
st.code(""" automationRisk_counts = new_df['Automation_Risk'].value_counts() """)
automationRisk_counts = new_df['Automation_Risk'].value_counts()
st.write(automationRisk_counts)

# Assuming your DataFrame is named 'weather_df_new'
selected_AutomationRisk = ["Medium", "High"]

# Filter the DataFrame to keep only the desired summaries
new_df_filtered = new_df[new_df['Automation_Risk'].isin(selected_AutomationRisk)]

# Reset the index if needed
new_df_filtered = new_df_filtered.reset_index(drop=True)

st.write("### new_df_filtered:")
st.code("""
       * selected_AutomationRisk = ["Medium", "High"]
       * new_df_filtered = new_df[new_df['Automation_Risk'].isin(selected_AutomationRisk)]
       * new_df_filtered = new_df_filtered.reset_index(drop=True)
""")

st.write(new_df_filtered.head())

st.write("### **Observation**")
st.write("We identified the counts of automationRisk, and filtered out the one with the lowest, which is Low with 158. After that, we created a new df with the name new_df_filtered having Automation_Risk with only Medium and High.")


AutomationRisk_counts = new_df_filtered['Automation_Risk'].value_counts()
st.code("""AutomationRisk_counts = new_df_filtered['Automation_Risk'].value_counts()""")
st.write(AutomationRisk_counts)


balanced_new_df = pd.DataFrame()
st.write("#### Initialize an empty dataframe to store balanced data")
st.code("balanced_new_df = pd.DataFrame()")


for Automation_Risk in AutomationRisk_counts.index:
    if Automation_Risk == 'Low':
        sampled_df = new_df_filtered[new_df_filtered['Automation_Risk'] == Automation_Risk]
    else:
        sampled_df = new_df_filtered[new_df_filtered['Automation_Risk'] == Automation_Risk].sample(169, random_state=42)
        
    balanced_new_df = pd.concat([balanced_new_df, sampled_df])
    
st.write("#### Loop through each category and sample: ")
st.code("""
for Automation_Risk in AutomationRisk_counts.index:
    if Automation_Risk == 'Low':
        sampled_df = new_df_filtered[new_df_filtered['Automation_Risk'] == Automation_Risk]
    else:
        sampled_df = new_df_filtered[new_df_filtered['Automation_Risk'] == Automation_Risk].sample(169, random_state=42)
        
    balanced_new_df = pd.concat([balanced_new_df, sampled_df])
""")

# Reset index if necessary
balanced_new_df.reset_index(drop=True, inplace=True)
st.write("#### Reset index: ")
st.code("balanced_new_df.reset_index(drop=True, inplace=True)")

st.write("#### Value count of Automation_risk: ")
st.write((balanced_new_df['Automation_Risk'].value_counts()))

balanced_new_df['Automation_Risk_encoded'] = automation_encoder.fit_transform(balanced_new_df['Automation_Risk'])

st.code("balanced_new_df['Automation_Risk_encoded'] = automation_encoder.fit_transform(balanced_new_df['Automation_Risk'])")
st.write(balanced_new_df.head())

balanced_new_df['Automation_Risk'].unique()
balanced_new_df['Automation_Risk_encoded'].unique()

st.write("#### Categories of Automation_Risk: ")
st.code("balanced_new_df['Automation_Risk'].unique()")
st.write(balanced_new_df['Automation_Risk'].unique())

st.write("#### Categories of Automation_Risk_encoded: ")
st.code("balanced_new_df['Automation_Risk_encoded'].unique()")
st.write(balanced_new_df['Automation_Risk_encoded'].unique())


balanced_unique_automationRisk = balanced_new_df['Automation_Risk'].unique()
balanced_unique_Automation_Risk_encoded = balanced_new_df['Automation_Risk_encoded'].unique()

balanced_automationRisk_mapping_df = pd.DataFrame({'Summary': balanced_unique_automationRisk, 'Summary_Encoded': balanced_unique_Automation_Risk_encoded})
balanced_automationRisk_mapping_df

st.write("### Code for Creating Automation Risk Mapping DataFrame")
st.code("""
# Mapping of the Summary and their encoded equivalent

balanced_unique_automationRisk = balanced_new_df['Automation_Risk'].unique()
balanced_unique_Automation_Risk_encoded = balanced_new_df['Automation_Risk_encoded'].unique()

# Create a new DataFrame
balanced_automationRisk_mapping_df = pd.DataFrame({
    'Summary': balanced_unique_automationRisk,
    'Summary_Encoded': balanced_unique_Automation_Risk_encoded
})
""")

# Display the DataFrame
st.write("### Automation Risk Mapping DataFrame")
st.dataframe(balanced_automationRisk_mapping_df)


balanced_new_df['Job_encoded'] = job_encoder.fit_transform(balanced_new_df['Job_Title'])
balanced_new_df['Industry_encoded'] = industry_encoder.fit_transform(balanced_new_df['Industry'])
balanced_new_df['Location_encoded'] = location_encoder.fit_transform(balanced_new_df['Location'])
balanced_new_df['Skills_encoded'] = skills_encoder.fit_transform(balanced_new_df['Required_Skills'])
balanced_new_df['Salary_USD'] = balanced_new_df['Salary_USD']

st.write("### Code for Encoding Categorical Columns and Converting Salary to USD")
st.code("""
balanced_new_df['Job_encoded'] = job_encoder.fit_transform(balanced_new_df['Job_Title'])
balanced_new_df['Industry_encoded'] = industry_encoder.fit_transform(balanced_new_df['Industry'])
balanced_new_df['Location_encoded'] = location_encoder.fit_transform(balanced_new_df['Location'])
balanced_new_df['Skills_encoded'] = skills_encoder.fit_transform(balanced_new_df['Required_Skills'])
balanced_new_df['Salary_USD'] = balanced_new_df['Salary_USD']
""")

# Select features and target variable
features = [ 'Salary_USD','Industry_encoded', 'Job_encoded', 'Location_encoded', 'Skills_encoded']
X1 = balanced_new_df[features]
Y1 = balanced_new_df['Automation_Risk_encoded']

st.write("### Code for Selecting Features and Target Variable")
st.code("""
# Select features and target variable
features = ['Salary_USD', 'Industry_encoded', 'Job_encoded', 'Location_encoded', 'Skills_encoded']
X1 = balanced_new_df[features]
Y1 = balanced_new_df['Automation_Risk_encoded']
""")

scaler = MinMaxScaler(feature_range=(0, 1))
num_col_ = [col for col in X1.columns if X1[col].dtype != 'object']
X1 = X1
X1[num_col_] = scaler.fit_transform(X1[num_col_])

st.write("### Code for Min-Max Scaling (Normalization)")
st.code("""
# Min-Max Scaling (Normalization)

# Initializing the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Selecting numerical columns
num_col_ = [col for col in X1.columns if X1[col].dtype != 'object']

# Learning the statistical parameters for each column and transforming
X1[num_col_] = scaler.fit_transform(X1[num_col_])
""")
st.write(X1.head())

st.write("### X1: ")
st.write(X1)

st.write("### Y1: ")
st.write(Y1)

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=42)

st.write("### Code for Splitting Data into Training and Testing Sets")
st.code("""
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=42)
""")

st.write("### **Observation**")
st.write("We do this so that we can split the new dataframe to Train and Test set.")

st.write('### Training Features (X1_train):')
st.dataframe(X1_train)

st.write('### Testing Features (X1_test):')
st.dataframe(X1_test)

st.write('### Training Target Variable (Y1_train):')
st.dataframe(Y1_train)

st.write('### Testing Target Variable (Y1_test):')
st.dataframe(Y1_test)

st.markdown("""
*  We used `.head()` to give us a preview of the `X1_test` and `X1_train` data.
*  We used `.shape` to give us the dimensions of the `X1_test` and `X1_train` data.
* Same case with the Y1_train and Y1_test.

 As you can see, the test shape has a lower number than the training shape. Since we should always allot more data to training for the machine learning model to practice.
""")

clf = RandomForestClassifier(random_state=42)
clf.fit(X1_train, Y1_train)

st.code("""
* clf = RandomForestClassifier(random_state=42)
* clf.fit(X1_train, Y1_train)""")

st.write("### **Observation**")
st.write("We initialized clf again for the new dataset that is used for the second training.")

st.subheader("Model Evaluation")

balanced_AutomationRisk_counts = balanced_new_df['Automation_Risk'].value_counts()
balanced_AutomationRisk_counts_list = balanced_AutomationRisk_counts.tolist()

st.code("""
* balanced_AutomationRisk_counts = balanced_new_df['Automation_Risk'].value_counts()
* balanced_AutomationRisk_counts_list = balanced_AutomationRisk_counts.tolist()
""")
st.write(balanced_AutomationRisk_counts_list)

balanced_AutomationRisk_list = balanced_new_df['Automation_Risk'].unique().tolist()

st.code("balanced_AutomationRisk_list = balanced_new_df['Automation_Risk'].unique().tolist()")
st.write('### Converting the .unique results to a list')
st.write(balanced_AutomationRisk_list)


Y1_pred = clf.predict(X1_test)
accuracy = accuracy_score(Y1_test, Y1_pred)

st.write('### Y1_pred and accuracy:')
st.code("""
* Y1_pred = clf.predict(X1_test
* accuracy = accuracy_score(Y1_test, Y1_pred)
""")
st.write(f"### Accuracy: {accuracy * 100:.2f}%")

feature_importance = clf.feature_importances_

st.write("### Feature_importance")
st.code("feature_importance = clf.feature_importances_")
st.write(feature_importance)

importance_df = pd.DataFrame({
    'Feature': X1.columns,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

st.code("""
        importance_df = pd.DataFrame({
    'Feature': X1.columns,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)""")

st.write(importance_df)
st.write("### **Observation**")
st.write("We used featured_importance for us to define the attribute that affect the machine training the most.")

plt.figure(figsize=(20, 10))
tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=balanced_AutomationRisk_list, rounded=True, proportion=True)

st.write("### Tree plot: ")
st.code("""plt.figure(figsize=(20, 10))
tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=balanced_AutomationRisk_list, rounded=True, proportion=True)""")
st.pyplot(plt)

st.write("### **Observation**")
st.write("We have shown the treeplot for the second training.")

###############################################################################################################################################################################################

#Growth Prediction
st.write("# Growth Prediction")

features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Salary_USD', 'Automation_encoded']
X2 = df_data[features]
Y2 = df_data['Growth_encoded']

st.subheader('Features (X2):')
st.write(X2)

st.subheader('Target Variable (Y2):')
st.write(Y2)

st.write("### **Observation**")
st.write(" To prepare for machine learning we categorized the needed datatype for predicition inside X. While we put the data will be predicted to Y which is Job_Growth_Projection.")

# Min-Max Scaling (Normalization)

scaler = MinMaxScaler(feature_range=(0, 1))

num_col_ = [col for col in X2.columns if X2[col].dtype != 'object']
X2 = X2

X2[num_col_] = scaler.fit_transform(X2[num_col_])

st.subheader('Min-Max Scaling (Normalization)')

st.write('Scaled Features (X2):')
st.dataframe(X2)

st.write("### **Observation**")
st.write(" The Min-Max scaling normalizes the numerical columns in X2 to a range of 0–1, making features comparable for machine learning models. The SettingWithCopyWarning suggests using .loc to ensure that the transformation applies directly to X2 without creating a copy.")

st.subheader('Job Growth Distribution')

jobGrowth_list = df['Job_Growth_Projection'].unique().tolist()
jobGrowth_counts = df['Job_Growth_Projection'].value_counts()
jobGrowth_counts_list = jobGrowth_counts.tolist()

# Get counts for the 'Job_Growth_Projection' column
jobGrowth_counts = df['Job_Growth_Projection'].value_counts()
jobGrowth_list = jobGrowth_counts.index
jobGrowth_counts_list = jobGrowth_counts.values

# Function to create a bar chart
def bar_chart_JobGrowth():
    colors = plt.cm.Paired(np.linspace(0, 1, len(jobGrowth_counts_list)))  # Creates different colors for each bar
    plt.bar(jobGrowth_list, jobGrowth_counts_list, color=colors)
    plt.xlabel('Job Growth Category')
    plt.ylabel('Count')
    plt.title('Bar Chart of Job Growth Projection')
    plt.xticks(rotation=0)
    st.pyplot(plt)

# Call the function to display the bar chart
bar_chart_JobGrowth()

st.write("### **Observation**")
st.write("We created a bar chart for our Growth prediction (our second Y) and we can see that it is almost evenly distributed. Which is good for our machine learning to not have any biases.")

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.1, random_state=42)

st.code("""X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.1, random_state=42)""")

st.write('Training Features (X2_train):')
st.dataframe(X2_train)

st.write('Testing Features (X2_test):')
st.dataframe(X2_test)

st.write('Training Target Variable (Y2_train):')
st.dataframe(Y2_train)

st.write('Testing Target Variable (Y2_test):')
st.dataframe(Y2_test)

st.write("### **Observation**")
st.markdown("""
*  We used `.head()` to give us a preview of the `X2_test` and `X2_train` data.
*  We used `.shape` to give us the dimensions of the `X2_test` and `X2_train` data.
* Same case with the Y2_train and Y2_test.

 As you can see, the test shape has a lower number than the training shape. Since we should always allot more data to training for the machine learning model to practice.
""")

clf = RandomForestClassifier(random_state=42)
clf.fit(X2_train, Y2_train)

st.code("""
* clf = RandomForestClassifier(random_state=42)
* clf.fit(X2_train, Y2_train)""")

st.write("RandomForestClassifier model trained successfully!")

st.write("### **Observation**")
st.write(" We used RandomForest Classifier algorithm for our supervised learning. To improve prediction accuracy and reduce overfitting.")

st.markdown("#### **Model Evaluation**")

Y2_pred = clf.predict(X2_test) #Prediction

st.code("""Y2_pred = clf.predict(X2_test) #Prediction""")

# Performance evaluation
accuracy = accuracy_score(Y2_test, Y2_pred)
classification_rep = classification_report(Y2_test, Y2_pred, target_names=growth_encoder.classes_)

st.write("### Classification Report")
st.text(classification_rep)

accuracy = accuracy_score(Y2_test, Y2_pred)
st.write("### Accuracy:")
st.markdown(f'{accuracy * 100:.2f}%')

st.write("### **Observation**")
st.write(" As we predict the Y and Based on the Performance evaluation - We can see the accuracy is low.")

feature_importance = clf.feature_importances_
st.code("""feature_importance = clf.feature_importances_""")
st.write("### feature_importance:")
st.write(feature_importance)

importance_df = pd.DataFrame({
    'Feature': X2.columns,
    'Importance': feature_importance
})

# Sort the DataFrame by importance for better readability
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
st.code("""importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)""")
# Display the resulting DataFrame
st.write("### importance_df:")
st.write(importance_df)

st.write("### **Observation**")
st.markdown("""
*  With the use of feature_importance we can identify each columns weighs in the model's capability of making decisions.
*  We can see from the results that Salary, Skills, Industry, Job Title, and Location weighs significantly more than the rest of the columns as we trained.
""")

fig = plt.figure(figsize=(20, 10))

st.write("### Tree plot:")
st.code("""
       * plt.figure(figsize=(20, 10))
       * tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=jobGrowth_list, rounded=True, proportion=True)
""")

tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=jobGrowth_list, rounded=True, proportion=True)
st.pyplot(fig)

st.write("### **Observation**")
st.write("The tree will help us visualize how the RFC decides.")

st.subheader("Training a better model.")
st.write("Our goal now is to balance out all of the classes.")

new_df = df
st.write("### New dataframe:")
st.code("""new_df = df""")
st.write(new_df.head())

st.write("### **Observation**")
st.write("We transfered df to  new_df.")


st.write("### JobGrowth_counts:")
st.code(""" jobGrowth_counts = new_df['Job_Growth_Projection'].value_counts() """)
jobGrowth_counts = new_df['Job_Growth_Projection'].value_counts()
st.write(jobGrowth_counts)


selected_jobGrowth = ["Growth", "Decline"]

new_df_filtered = new_df[new_df['Job_Growth_Projection'].isin(selected_jobGrowth)]

# Reset the index if needed
new_df_filtered = new_df_filtered.reset_index(drop=True)

st.write("### new_df_filtered:")
st.code("""
       * selected_jobGrowth = ["Growth", "Decline"]
       * new_df_filtered = new_df[new_df['Job_Growth_Projection'].isin(selected_jobGrowth)]
       * new_df_filtered = new_df_filtered.reset_index(drop=True)
""")

st.write(new_df_filtered.head())

st.write("### **Observation**")
st.write("We identified the counts of jobGrowth, and filtered out the one with the lowest, which is Decline with 169. After that, we created a new df with the name new_df_filtered having Automation_Risk with only Growth and Decline.")


jobGrowth_counts = new_df_filtered['Job_Growth_Projection'].value_counts()
st.code("""jobGrowth_counts = new_df_filtered['Job_Growth_Projection'].value_counts()""")
st.write(jobGrowth_counts)


balanced_new_df = pd.DataFrame()
st.write("#### Initialize an empty dataframe to store balanced data")
st.code("balanced_new_df = pd.DataFrame()")


for Job_Growth_Projection in jobGrowth_counts.index:
    if Job_Growth_Projection == 'Stable':
        sampled_df = new_df_filtered[new_df_filtered['Job_Growth_Projection'] == Job_Growth_Projection]
    else:
        sampled_df = new_df_filtered[new_df_filtered['Job_Growth_Projection'] == Job_Growth_Projection].sample(169, random_state=42)

    
    balanced_new_df = pd.concat([balanced_new_df, sampled_df])
    
st.write("#### Loop through each category and sample: ")
st.code("""
for Job_Growth_Projection in jobGrowth_counts.index:
    if Job_Growth_Projection == 'Stable':
        sampled_df = new_df_filtered[new_df_filtered['Job_Growth_Projection'] == Job_Growth_Projection]
    else:
        sampled_df = new_df_filtered[new_df_filtered['Job_Growth_Projection'] == Job_Growth_Projection].sample(169, random_state=42)

    
    balanced_new_df = pd.concat([balanced_new_df, sampled_df])
""")

# Reset index if necessary
balanced_new_df.reset_index(drop=True, inplace=True)
st.write("#### Reset index: ")
st.code("balanced_new_df.reset_index(drop=True, inplace=True)")

st.write("#### Value count of Job Growth: ")
st.write((balanced_new_df['Job_Growth_Projection'].value_counts()))

balanced_new_df['Growth_encoded'] = growth_encoder.fit_transform(balanced_new_df['Job_Growth_Projection'])

st.code("balanced_new_df['Growth_encoded'] = growth_encoder.fit_transform(balanced_new_df['Job_Growth_Projection'])")
st.write(balanced_new_df.head())

balanced_new_df['Job_Growth_Projection'].unique()
balanced_new_df['Growth_encoded'].unique()

st.write("#### Categories of Growth Prediction: ")
st.code("balanced_unique_jobGrowth = balanced_new_df['Job_Growth_Projection'].unique()")
st.write(balanced_unique_jobGrowth = balanced_new_df['Job_Growth_Projection'].unique())

st.write("#### Categories of Growth_encoded: ")
st.code("balanced_new_df['Growth_encoded'].unique()")
st.write(balanced_new_df['Growth_encoded'].unique())


balanced_unique_jobGrowth = balanced_new_df['Job_Growth_Projection'].unique()
balanced_unique_Growth_encoded = balanced_new_df['Growth_encoded'].unique()

balanced_jobGrowth_mapping_df = pd.DataFrame({'Summary': balanced_unique_jobGrowth, 'Summary_Encoded': balanced_unique_Growth_encoded})
balanced_jobGrowth_mapping_df

st.write("### Code for Creating JobGrowth Mapping DataFrame")
st.code("""
# Mapping of the Summary and their encoded equivalent

balanced_unique_jobGrowth = balanced_new_df['Job_Growth_Projection'].unique()
balanced_unique_Growth_encoded = balanced_new_df['Growth_encoded'].unique()

# Create a new DataFrame
balanced_jobGrowth_mapping_df = pd.DataFrame({'Summary': balanced_unique_jobGrowth, 'Summary_Encoded': balanced_unique_Growth_encoded})
})
""")

# Display the DataFrame
st.write("### Growth prediction Mapping DataFrame")
st.dataframe(balanced_jobGrowth_mapping_df)


balanced_new_df['Job_encoded'] = job_encoder.fit_transform(balanced_new_df['Job_Title'])
balanced_new_df['Industry_encoded'] = industry_encoder.fit_transform(balanced_new_df['Industry'])
balanced_new_df['Location_encoded'] = location_encoder.fit_transform(balanced_new_df['Location'])
balanced_new_df['Skills_encoded'] = skills_encoder.fit_transform(balanced_new_df['Required_Skills'])
balanced_new_df['Salary_USD'] = balanced_new_df['Salary_USD']

st.write("### Code for Encoding Categorical Columns and Converting Salary to USD")
st.code("""
balanced_new_df['Job_encoded'] = job_encoder.fit_transform(balanced_new_df['Job_Title'])
balanced_new_df['Industry_encoded'] = industry_encoder.fit_transform(balanced_new_df['Industry'])
balanced_new_df['Location_encoded'] = location_encoder.fit_transform(balanced_new_df['Location'])
balanced_new_df['Skills_encoded'] = skills_encoder.fit_transform(balanced_new_df['Required_Skills'])
balanced_new_df['Salary_USD'] = balanced_new_df['Salary_USD']
""")

# Select features and target variable
features = [ 'Salary_USD', 'Industry_encoded', 'Job_encoded', 'Location_encoded', 'Skills_encoded']
X2 = balanced_new_df[features]
Y2 = balanced_new_df['Growth_encoded']

st.write("### Code for Selecting Features and Target Variable")
st.code("""
# Select features and target variable
features = [ 'Salary_USD', 'Industry_encoded', 'Job_encoded', 'Location_encoded', 'Skills_encoded']
X2 = balanced_new_df[features]
Y2 = balanced_new_df['Growth_encoded']
""")

scaler = MinMaxScaler(feature_range=(0, 1))
num_col_ = [col for col in X2.columns if X2[col].dtype != 'object']
X2 = X2
X2[num_col_] = scaler.fit_transform(X2[num_col_])

st.write("### Code for Min-Max Scaling (Normalization)")
st.code("""
# Min-Max Scaling (Normalization)

# Initializing the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Selecting numerical columns
num_col_ = [col for col in X2.columns if X2[col].dtype != 'object']

# Learning the statistical parameters for each column and transforming
X2[num_col_] = scaler.fit_transform(X2[num_col_])
""")
st.write(X2.head())

st.write("### X2: ")
st.write(X2)

st.write("### Y2: ")
st.write(Y2)

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.3, random_state=42)

st.write("### Code for Splitting Data into Training and Testing Sets")
st.code("""
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.3, random_state=42)
""")

st.write("### **Observation**")
st.write("We do this so that we can split the new dataframe to Train and Test set.")

st.write('### Training Features (X2_train):')
st.dataframe(X2_train)

st.write('### Testing Features (X2_test):')
st.dataframe(X2_test)

st.write('### Training Target Variable (Y2_train):')
st.dataframe(Y2_train)

st.write('### Testing Target Variable (Y2_test):')
st.dataframe(Y2_test)

st.markdown("""
*  We used `.head()` to give us a preview of the `X2_test` and `X2_train` data.
*  We used `.shape` to give us the dimensions of the `X2_test` and `X2_train` data.
* Same case with the Y2_train and Y2_test.

 As you can see, the test shape has a lower number than the training shape. Since we should always allot more data to training for the machine learning model to practice.
""")

clf = RandomForestClassifier(random_state=42)
clf.fit(X2_train, Y2_train)

st.code("""
* clf = RandomForestClassifier(random_state=42)
* clf.fit(X2_train, Y2_train)""")

st.write("### **Observation**")
st.write("We initialized clf again for the new dataset that is used for the second training.")

st.subheader("Model Evaluation")

balanced_jobGrowth_counts = balanced_new_df['Job_Growth_Projection'].value_counts()
balanced_jobGrowth_counts_list = balanced_jobGrowth_counts.tolist()

st.code("""
* balanced_jobGrowth_counts = balanced_new_df['Job_Growth_Projection'].value_counts()
* balanced_jobGrowth_counts_list = balanced_jobGrowth_counts.tolist()
""")
st.write(balanced_jobGrowth_counts_list)

balanced_jobGrowth_list = balanced_new_df['Job_Growth_Projection'].unique().tolist()

st.code("balanced_jobGrowth_list = balanced_new_df['Job_Growth_Projection'].unique().tolist()")
st.write('### Converting the .unique results to a list')
st.write(balanced_jobGrowth_list)


Y2_pred = clf.predict(X2_test)
accuracy = accuracy_score(Y2_test, Y2_pred)

st.write('### Y2_pred and accuracy:')
st.code("""
* Y2_pred = clf.predict(X1_test
* accuracy = accuracy_score(Y2_test, Y2_pred)
""")
st.write(f"### Accuracy: {accuracy * 100:.2f}%")

feature_importance = clf.feature_importances_

st.write("### Feature_importance")
st.code("feature_importance = clf.feature_importances_")
st.write(feature_importance)

importance_df = pd.DataFrame({
    'Feature': X2.columns,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

st.code("""
        importance_df = pd.DataFrame({
    'Feature': X2.columns,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)""")

st.write(importance_df)
st.write("### **Observation**")
st.write("We used featured_importance for us to define the attribute that affect the machine training the most.")

plt.figure(figsize=(20, 10))
tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=balanced_jobGrowth_list, rounded=True, proportion=True)

st.write("### Tree plot: ")
st.code("""plt.figure(figsize=(20, 10))
tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=balanced_jobGrowth_list, rounded=True, proportion=True)""")
st.pyplot(plt)

st.write("### **Observation**")
st.write("We have shown the treeplot for the second training.")

##############################################################################################################################################################################################

#Salary Prediction
st.write("# Salary Prediction")

def categorize_salary(salary):
    if salary < 50000:
        return 'Entry Level'
    elif salary < 100000:
        return 'Mid Level'
    elif salary < 200000:
        return 'Senior Level'

df['Salary_Category'] = df['Salary_USD'].apply(categorize_salary)

st.code("""def categorize_salary(salary):
    if salary < 50000:
        return 'Entry Level'
    elif salary < 100000:
        return 'Mid Level'
    elif salary < 200000:
        return 'Senior Level'

df['Salary_Category'] = df['Salary_USD'].apply(categorize_salary)""")

st.write("### **Observation**")
st.write("This categorizes salaries to 3 categories, Entry Level, Mid Level, and Senior Level, this allows us to simplify the salary data and makes it easier to identify.")

df = df.drop(columns=['Salary_USD'])
df_data = df_data.drop(columns=['Salary_USD'])

st.code("""df = df.drop(columns=['Salary_USD'])
df_data = df_data.drop(columns=['Salary_USD'])""")

st.write("### **Observation**")
st.write("We remove the column Salary_USD from df and df_data.")

salary_encoder = LabelEncoder()
df_data['Salary_encoded'] = salary_encoder.fit_transform(df['Salary_Category'])

st.code("""salary_encoder = LabelEncoder()
df_data['Salary_encoded'] = salary_encoder.fit_transform(df['Salary_Category'])""")

st.write("### **Observation**")
st.write("Encoded the Salary_Category..")

features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Automation_encoded', 'Growth_encoded']
X4 = df_data[features]
Y4 = df_data['Salary_encoded']

st.subheader('Features (X4):')
st.write(X4)

st.subheader('Target Variable (Y4):')
st.write(Y4)

st.write("### **Observation**")
st.write(" To prepare for machine learning we categorized the needed datatype for predicition inside X. While we put the data will be predicted to Y which is Salary_Category")

# Min-Max Scaling (Normalization)

scaler = MinMaxScaler(feature_range=(0, 1))

num_col_ = [col for col in X4.columns if X4[col].dtype != 'object']
X4 = X4

X4[num_col_] = scaler.fit_transform(X4[num_col_])

st.subheader('Min-Max Scaling (Normalization)')

st.write('Scaled Features (X4):')
st.dataframe(X4)

st.write("### **Observation**")
st.write(" All numerical values were rescaled to a 0–1 range using min-max scaling, which made sure that no feature was overpowered by scale discrepancies. This increases the effectiveness of model training and produces predictions that are more trustworthy.")

st.subheader('Salary Category Distribution')

salaryCategory_counts = df['Salary_Category'].value_counts()
salaryCategory_list = salaryCategory_counts.index
salaryCategory_counts_list = salaryCategory_counts.values

def bar_chart_SalaryCategory():
    plt.clf()
    colors = plt.cm.Paired(np.linspace(0, 1, len(salaryCategory_counts_list)))  # Creates different colors for each bar
    plt.bar(salaryCategory_list, salaryCategory_counts_list, color=colors)
    plt.xlabel('Salary Category')
    plt.ylabel('Count')
    plt.title('Bar Chart of Salary Category')
    plt.xticks(rotation=0)
    
bar_chart_SalaryCategory()
st.pyplot(plt)

st.write("### **Observation**")
st.write("The chart shows that Mid Level has the highest count, followed by the Senior Leve, and Entry level being the lowest. This shows that there is a significant distribution between the data.")

X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y4, test_size=0.3, random_state=42)

st.code("""X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y4, test_size=0.3, random_state=42)""")

st.write('Training Features (X4_train):')
st.dataframe(X4_train)

st.write('Testing Features (X4_test):')
st.dataframe(X4_test)

st.write('Training Target Variable (Y4_train):')
st.dataframe(Y4_train)

st.write('Testing Target Variable (Y4_test):')
st.dataframe(Y4_test)

st.write("### **Observation**")
st.markdown("""
*  We used `.head()` to give us a preview of the `X4_test` and `X4_train` data.
*  We used `.shape` to give us the dimensions of the `X4_test` and `X4_train` data.
* Same case with the Y4_train and Y4_test.

 As you can see, the test shape has a lower number than the training shape. Since we should always allot more data to training for the machine learning model to practice.
""")

clf = RandomForestClassifier(random_state=42)
clf.fit(X4_train, Y4_train)

st.code("""
* clf = RandomForestClassifier(random_state=42)
* clf.fit(X4_train, Y4_train)""")

st.write("RandomForestClassifier model trained successfully!")

st.write("### **Observation**")
st.write(" We used RandomForest Classifier algorithm for our supervised learning. To improve prediction accuracy and reduce overfitting.")

st.markdown("#### **Model Evaluation**")

Y4_pred = clf.predict(X4_test) #Prediction

st.code("""Y4_pred = clf.predict(X4_test) #Prediction""")

# Performance evaluation
accuracy = accuracy_score(Y4_test, Y4_pred)
classification_rep = classification_report(Y4_test, Y4_pred, target_names=salary_encoder.classes_)

st.write("### Classification Report")
st.text(classification_rep)

accuracy = accuracy_score(Y4_test, Y4_pred)
st.write("### Accuracy:")
st.markdown(f'{accuracy * 100:.2f}%')

st.write("### **Observation**")
st.write(" After testing,we can see that the Accuracy is average, the accuracy of the model in predicting.")

feature_importance = clf.feature_importances_
st.code("""feature_importance = clf.feature_importances_""")
st.write("### feature_importance:")
st.write(feature_importance)

importance_df = pd.DataFrame({
    'Feature': X4.columns,
    'Importance': feature_importance
})

# Sort the DataFrame by importance for better readability
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
st.code("""importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)""")
# Display the resulting DataFrame
st.write("### importance_df:")
st.write(importance_df)

st.write("### **Observation**")
st.markdown("""
The data presents the importance of different features of the model. The data shows that Skills_encoded is important in predicting the salary category, followd by Job_encoded and Industry_encoded.
""")

fig = plt.figure(figsize=(20, 10))

st.write("### Tree plot:")
st.code("""
       * plt.figure(figsize=(20, 10))
       * tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=salaryCategory_list, rounded=True, proportion=True)
""")

tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=salaryCategory_list, rounded=True, proportion=True)
st.pyplot(fig)

st.write("### **Observation**")
st.write("The tree will help us visualize how the RFC decides.")

st.subheader("Training a better model.")
st.write("Our goal now is to balance out all of the classes.")

new_df = df
st.write("### New dataframe:")
st.code("""new_df = df""")
st.write(new_df.head())

st.write("### **Observation**")
st.write("We transfered df to  new_df.")


st.write("### salaryCategory_counts:")
st.code(""" salaryCategory_counts = new_df['Salary_Category'].value_counts() """)
salaryCategory_counts = new_df['Salary_Category'].value_counts()
st.write(salaryCategory_counts)


selected_salaryCategory = ["Mid Level", "Senior Level"]

new_df_filtered = new_df[new_df['Salary_Category'].isin(selected_salaryCategory)]

# Reset the index if needed
new_df_filtered = new_df_filtered.reset_index(drop=True)

st.write("### new_df_filtered:")
st.code("""
       * selected_salaryCategory = ["Mid Level", "Senior Level"]
       * new_df_filtered = new_df[new_df['Salary_Category'].isin(selected_salaryCategory)]
       * new_df_filtered = new_df_filtered.reset_index(drop=True)
""")

st.write(new_df_filtered.head())

st.write("### **Observation**")
st.write("We identified the counts of salary category, and filtered out the one with the lowest, which is Senior level with 172. After that, we created a new df with the name new_df_filtered having Salary_cateogry with only Mid Level and Senior Level.")


salaryCategory_counts = new_df['Salary_Category'].value_counts()
st.code("""salaryCategory_counts = new_df['Salary_Category'].value_counts()""")
st.write(salaryCategory_counts)


balanced_new_df = pd.DataFrame()
st.write("#### Initialize an empty dataframe to store balanced data")
st.code("balanced_new_df = pd.DataFrame()")


for Salary_Category in salaryCategory_counts.index:
    if Salary_Category  == 'Entry Level':
        sampled_df = new_df_filtered[new_df_filtered['Salary_Category'] == Salary_Category ]
    else:
        sampled_df = new_df_filtered[new_df_filtered['Salary_Category'] == Salary_Category ].sample(172, random_state=42)

    balanced_new_df = pd.concat([balanced_new_df, sampled_df])
    
st.write("#### Loop through each category and sample: ")
st.code("""
for Salary_Category  in salaryCategory_counts.index:
    if Salary_Category  == 'Entry Level':
        sampled_df = new_df_filtered[new_df_filtered['Salary_Category'] == Salary_Category ]
    else:
        sampled_df = new_df_filtered[new_df_filtered['Salary_Category'] == Salary_Category ].sample(172, random_state=42)

    balanced_new_df = pd.concat([balanced_new_df, sampled_df])
""")

# Reset index if necessary
balanced_new_df.reset_index(drop=True, inplace=True)
st.write("#### Reset index: ")
st.code("balanced_new_df.reset_index(drop=True, inplace=True)")

st.write("#### Value count of Job Growth: ")
st.write((balanced_new_df['Salary_Category'].value_counts()))

balanced_new_df['Salary_encoded'] = salary_encoder.fit_transform(balanced_new_df['Salary_Category'])

st.code("balanced_new_df['Salary_encoded'] = growth_encoder.fit_transform(balanced_new_df['Salary_Category'])")
st.write(balanced_new_df.head())

balanced_new_df['Salary_Category'].unique()
balanced_new_df['Salary_encoded'].unique()

st.write("#### Categories of Salary Category Prediction: ")
st.code("balanced_unique_salaryCategory = balanced_new_df['Salary_Category'].unique()")
st.write(balanced_unique_salaryCategory = balanced_new_df['Salary_Category'].unique())

st.write("#### Categories of Growth_encoded: ")
st.code("balanced_new_df['Salary_encoded'].unique()")
st.write(balanced_new_df['Salary_encoded'].unique())

balanced_unique_salaryCategory = balanced_new_df['Salary_Category'].unique()
balanced_unique_salaryCategory_encoded = balanced_new_df['Salary_encoded'].unique()

balanced_salaryCategory_mapping_df = pd.DataFrame({'Summary': balanced_unique_salaryCategory, 'Summary_Encoded': balanced_unique_salaryCategory_encoded})
balanced_salaryCategory_mapping_df

st.write("### Code for Creating JobGrowth Mapping DataFrame")
st.code("""
# Mapping of the Summary and their encoded equivalent

balanced_unique_salaryCategory = balanced_new_df['Salary_Category'].unique()
balanced_unique_salaryCategory_encoded = balanced_new_df['Salary_encoded'].unique()

# Create a new DataFrame
balanced_salaryCategory_mapping_df = pd.DataFrame({'Summary': balanced_unique_salaryCategory, 'Summary_Encoded': balanced_unique_salaryCategory_encoded})
""")

# Display the DataFrame
st.write("### Salary Category prediction Mapping DataFrame")
st.dataframe(balanced_salaryCategory_mapping_df)

balanced_new_df['Job_encoded'] = job_encoder.fit_transform(balanced_new_df['Job_Title'])
balanced_new_df['Industry_encoded'] = industry_encoder.fit_transform(balanced_new_df['Industry'])
balanced_new_df['Location_encoded'] = location_encoder.fit_transform(balanced_new_df['Location'])
balanced_new_df['Skills_encoded'] = skills_encoder.fit_transform(balanced_new_df['Required_Skills'])

st.write("### Code for Encoding Categorical Columns")
st.code("""
balanced_new_df['Job_encoded'] = job_encoder.fit_transform(balanced_new_df['Job_Title'])
balanced_new_df['Industry_encoded'] = industry_encoder.fit_transform(balanced_new_df['Industry'])
balanced_new_df['Location_encoded'] = location_encoder.fit_transform(balanced_new_df['Location'])
balanced_new_df['Skills_encoded'] = skills_encoder.fit_transform(balanced_new_df['Required_Skills'])
""")

# Select features and target variable
features = ['Job_encoded', 'Industry_encoded', 'Location_encoded', 'Skills_encoded']
X4 = balanced_new_df[features]
Y4 = balanced_new_df['Salary_encoded']

st.write("### Code for Selecting Features and Target Variable")
st.code("""
# Select features and target variable
features = ['Job_encoded', 'Industry_encoded', 'Location_encoded', 'Skills_encoded']
X4 = balanced_new_df[features]
Y4 = balanced_new_df['Salary_encoded']
""")

scaler = MinMaxScaler(feature_range=(0, 1))
num_col_ = [col for col in X4.columns if X4[col].dtype != 'object']
X4 = X4
X4[num_col_] = scaler.fit_transform(X4[num_col_])

st.write("### Code for Min-Max Scaling (Normalization)")
st.code("""
# Min-Max Scaling (Normalization)

# Initializing the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Selecting numerical columns
num_col_ = [col for col in X4.columns if X4[col].dtype != 'object']

# Learning the statistical parameters for each column and transforming
X4[num_col_] = scaler.fit_transform(X4[num_col_])
""")
st.write(X4.head())

st.write("### X4: ")
st.write(X4)

st.write("### Y2: ")
st.write(Y4)

X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y4, test_size=0.3, random_state=42)

st.write("### Code for Splitting Data into Training and Testing Sets")
st.code("""
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y4, test_size=0.3, random_state=42)
""")

st.write("### **Observation**")
st.write("We do this so that we can split the new dataframe to Train and Test set.")

st.write('### Training Features (X4_train):')
st.dataframe(X4_train)

st.write('### Testing Features (X4_test):')
st.dataframe(X4_test)

st.write('### Training Target Variable (Y4_train):')
st.dataframe(Y4_train)

st.write('### Testing Target Variable (Y4_test):')
st.dataframe(Y4_test)

st.markdown("""
*  Both datasets show a systematic approach to data preparation for machine learning, with encoded representations of job type, industry, location, and necessary abilities. Effective training and evaluation of models depend on this consistent feature set, which guarantees that the models can generalize well to new data.
*  The binary values for both target variables are encoded wage categories, with '1' most likely denoting a certain salary level (such as "Mid Level" or "Senior Level") and '0' denoting another level (such as "Entry Level"). Effective model training and evaluation are made possible by this binary encoding, which also makes classification problems involving the prediction of wage categories based on the features in X4_train and X4_test easier.
""")

clf = RandomForestClassifier(random_state=42)
clf.fit(X4_train, Y4_train)

st.code("""
* clf = RandomForestClassifier(random_state=42)
* clf.fit(X4_train, Y4_train)""")

st.write("### **Observation**")
st.write("We initialized clf again for the new dataset that is used for the second training.")

st.subheader("Model Evaluation")

balanced_salaryCategory_counts = balanced_new_df['Salary_Category'].value_counts()
balanced_salaryCategory_counts_list = balanced_salaryCategory_counts.tolist()

st.code("""
* balanced_salaryCategory_counts = balanced_new_df['Salary_Category'].value_counts()
* balanced_salaryCategory_counts_list = balanced_salaryCategory_counts.tolist()
""")
st.write(balanced_salaryCategory_counts_list)

balanced_salaryCategory_list = balanced_new_df['Salary_Category'].unique().tolist()

st.code("balanced_salaryCategory_list = balanced_new_df['Salary_Category'].unique().tolist()")
st.write('### Converting the .unique results to a list')
st.write(balanced_salaryCategory_list)


Y4_pred = clf.predict(X4_test)
accuracy = accuracy_score(Y4_test, Y4_pred)

st.write('### Y4_pred and accuracy:')
st.code("""
* Y4_pred = clf.predict(X4_test
* accuracy = accuracy_score(Y4_test, Y4_pred)
""")
st.write(f"### Accuracy: {accuracy * 100:.2f}%")

feature_importance = clf.feature_importances_

st.write("### Feature_importance")
st.code("feature_importance = clf.feature_importances_")
st.write(feature_importance)

importance_df = pd.DataFrame({
    'Feature': X4.columns,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

st.code("""
        importance_df = pd.DataFrame({
    'Feature': X4.columns,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)""")

st.write(importance_df)
st.write("### **Observation**")
st.write("We used featured_importance for us to define the attribute that affect the machine training the most.")

plt.figure(figsize=(20, 10))
tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=balanced_salaryCategory_list, rounded=True, proportion=True)

st.write("### Tree plot: ")
st.code("""plt.figure(figsize=(20, 10))
tree.plot_tree(clf.estimators_[0], filled=True, feature_names=features, class_names=balanced_salaryCategory_list, rounded=True, proportion=True)""")
st.pyplot(plt)

st.write("### **Observation**")
st.write("By doing the second training we were able to increase the accuracy of the program. This was possible by adding an importance analysis, by doing this it was able to focus on what was the most important aspect and was able to predict better. Adding the Visual Analysis also allows the program to refine the features and adjust the model parameters.")