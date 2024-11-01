#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

#######################
# Page configuration
st.set_page_config(
    page_title="Dashboard Template", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('🤖 AI-Powered Job Market Insight')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Turgo, Andrei Bernard\n2. Torres, Alistair Aaron\n3. Dantes, Nikkos Adrielle\n4. Solis, Jaymar\n5. Borrinaga, Don Carlo")

#######################
# Data

# Load data
dataset = pd.read_csv("data/AI.csv")

# Modifications

def categorize_salary(salary):
    if salary < 50000:
        return 'Entry Level'
    elif salary < 100000:
        return 'Mid Level'
    elif salary < 200000:
        return 'Senior Level'

dataset['Salary_Category'] = dataset['Salary_USD'].apply(categorize_salary)

#######################

#Pie Chart Automation Risk
automationRisk_list = dataset['Automation_Risk'].unique().tolist()
automationRisk_counts = dataset['Automation_Risk'].value_counts()
automationRisk_counts_list = automationRisk_counts.tolist()

#Bar Chart Job Growth Projection
jobGrowth_list = dataset['Job_Growth_Projection'].unique().tolist()
jobGrowth_counts = dataset['Job_Growth_Projection'].value_counts()
jobGrowth_counts_list = jobGrowth_counts.tolist()

#Bar Chart Salary Category
salaryCategory_list = dataset['Salary_Category'].unique().tolist()
salaryCategory_counts = dataset['Salary_Category'].value_counts()
salaryCategory_counts_list = salaryCategory_counts.tolist()


#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here
    st.markdown(""" 
    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to classify **Salary_USD**, **Automation_Risk**, and **Job_Growth_Projection** from the AI-Powered Job Market Insights dataset using **Random Forest Classifier**.

    #### Pages
    1. `Dataset` - Brief description of the AI-Powered Job Market Insights dataset used in this dashboard.
    2. `EDA` - Exploratory Data Analysis of the AI-Powered Job Market Insights dataset, highlighting the distribution of **Salary_USD**, **Automation_Risk**, and **Job_Growth_Projection**. This section includes graphs such as Pie Charts and Bar Charts.
    3. `Data Cleaning / Pre-processing` - Overview of data cleaning and pre-processing steps, including encoding for **Salary_USD**, **Automation_Risk**, and **Job_Growth_Projection** columns, as well as splitting the dataset into training and testing sets.
    4. `Machine Learning` - Training a supervised classification model using the **Random Forest Classifier**. This section covers model evaluation, feature importance, and tree plotting.
    5. `Prediction` - A prediction page where users can input values to predict **Salary**, **Automation Risk**, and **Job Growth Projection** using the trained models.
    6. `Conclusion` - A summary of the insights and observations from the EDA and model training.
""")

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("AI-Powered Job Market Insights Dataset")
    st.markdown("""

    The AI-Powered Job Market Insights dataset was introduced by Laksika Tharmalingam, it is a dataset provides a synthetic but realistic snapshot of the modern job market, particularly focusing on the role of artificial intelligence (AI) and automation across various industries. This dataset includes 500 unique job listings, each characterized by different factors like industry, company size, AI adoption level, automation risk, required skills, and job growth projections. It is designed to be a valuable resource for researchers, data scientists, and policymakers exploring the impact of AI on employment, job market trends, and the future of work.

    Content
    The dataset has 500 rows. The columns are as follows: Job_Title, Industry, Company_Size, Location, AI_Adoption_Level, Automation_Risk, Required_Skills, Salary_USD, Remote_Friendly, and Job_Growth_Projection.

    `Link:` https://www.kaggle.com/datasets/uom190346a/ai-powered-job-market-insights?fbclid=IwZXh0bgNhZW0CMTEAAR3rYnERo-UcWo7EcLpdyvZhk4s4CwVmkCm3QA-ifo347G-H57xxj-r3onA_aem_pLURKK0jybKoBp3Oxe8vpA

    """)

    # Your content for your DATASET page goes here
    
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(dataset, use_container_width=True, hide_index=True)
    
    st.subheader("Descriptive Statistics")
    st.dataframe(dataset.describe(), use_container_width=True)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")
    
    # Display the legend in an expander above the columns
    with st.expander('Legend', expanded=True):
        st.write('''
            - **Data**: [AI-Powered Job Market Insights Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset).
            - :green[Pie Chart]: Distribution of the Automation_Risk in the dataset.
            - :green[Bar Chart]: Distribution of the Job_Growth_Projection in the dataset.
            - :green[Bar Chart]: Distribution of the Salary_Category in the dataset.
        ''')

    col = st.columns((2, 2, 2), gap='small')

    # Automation Risk Distribution Pie Chart
    with col[0]:
        st.markdown('#### Automation Risk Distribution')
        def pie_chart_AutomationRisk():
            plt.pie(automationRisk_counts_list, labels=automationRisk_list, autopct='%1.1f%%')
            plt.title('Pie Chart of Automation Risk')
            st.pyplot(plt)
        pie_chart_AutomationRisk()
        
    # Job Growth Distribution Bar Chart
    with col[1]:
        st.markdown('#### Job Growth Distribution')
        def bar_chart_JobGrowth():
            plt.clf() 
            colors = plt.cm.Paired(np.linspace(0, 1, len(jobGrowth_counts_list)))  
            plt.bar(jobGrowth_list, jobGrowth_counts_list, color=colors)
            plt.xlabel('Job Growth Category')
            plt.ylabel('Count')
            plt.title('Bar Chart of Job Growth Projection')
            plt.xticks(rotation=0)
            st.pyplot(plt)
        bar_chart_JobGrowth()

    # Salary Category Distribution Bar Chart
    with col[2]:
        st.markdown('#### Salary Category Distribution')
        def bar_chart_SalaryCategory():
            plt.clf() 
            colors = plt.cm.Paired(np.linspace(0, 1, len(salaryCategory_counts_list))) 
            plt.bar(salaryCategory_list, salaryCategory_counts_list, color=colors)
            plt.xlabel('Salary Category')
            plt.ylabel('Count')
            plt.title('Bar Chart of Salary Category')
            plt.xticks(rotation=0)
            st.pyplot(plt)
        bar_chart_SalaryCategory()
        
     # Insights Section
    st.header("💡 Insights")
    
# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here