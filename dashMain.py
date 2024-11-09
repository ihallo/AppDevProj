#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn.metrics import accuracy_score
import joblib

#######################
# Page configuration
st.set_page_config(
    page_title="Streamlit-AI Job Market Insight - Dashboard", # Replace this with your Project's Title
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

# Importing models
clf_automation = joblib.load('models/RFC_Automation.joblib')
clf_growthPrediction = joblib.load('models/RFC_GrowthPrediction.joblib')
clf_salary = joblib.load('models/RFC_Salary.joblib')

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

def scatter_plot_Automation(column, width, height, key):

    # Generate a scatter plot
    scatter_plot = px.scatter(dataset, x=dataset['Automation_Risk'], y=dataset[column])

    # Adjust the height and width
    scatter_plot.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(scatter_plot, use_container_width=True, key=f"auto_scatter_plot_{key}")

def scatter_plot_Growth(column, width, height, key):

    # Generate a scatter plot
    scatter_plot = px.scatter(dataset, x=dataset['Job_Growth_Projection'], y=dataset[column])

    # Adjust the height and width
    scatter_plot.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(scatter_plot, use_container_width=True, key=f"growth_scatter_plot_{key}")
    
def scatter_plot_Salary(column, width, height, key):

    # Generate a scatter plot
    scatter_plot = px.scatter(dataset, x=dataset['Salary_Category'], y=dataset[column])

    # Adjust the height and width
    scatter_plot.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(scatter_plot, use_container_width=True, key=f"salary_scatter_plot_{key}")


##df_data

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
salary_encoder = LabelEncoder()

df_data['Job_encoded'] = job_encoder.fit_transform(dataset['Job_Title'])
df_data['Industry_encoded'] = industry_encoder.fit_transform(dataset['Industry'])
df_data['Size_encoded'] = size_encoder.fit_transform(dataset['Company_Size'])
df_data['Location_encoded'] = location_encoder.fit_transform(dataset['Location'])
df_data['AI_Adoption_encoded'] = ai_adoption_encoder.fit_transform(dataset['AI_Adoption_Level'])
df_data['Automation_encoded'] = automation_encoder.fit_transform(dataset['Automation_Risk'])
df_data['Skills_encoded'] = skills_encoder.fit_transform(dataset['Required_Skills'])
df_data['Remote_encoded'] = remote_encoder.fit_transform(dataset['Remote_Friendly'])
df_data['Growth_encoded'] = growth_encoder.fit_transform(dataset['Job_Growth_Projection'])
df_data['Salary_encoded'] = salary_encoder.fit_transform(dataset['Salary_Category'])
df_data['Salary_USD'] = dataset['Salary_USD']

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
    
    #Column Description
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
    
    st.subheader("Descriptive Statistics")
    st.dataframe(dataset.describe(), use_container_width=True)
    st.info("The dataset includes 500 salary data points with an average salary of USD 91,222.39. Salaries range from a minimum of USD 31,969.53 to a maximum of USD 155,209.82. The standard deviation is USD 20,504.29, indicating moderate variability. The median salary is USD 91,998.20, with 25% of salaries below USD 78,511.51 and 75% below USD 103,971.28.")

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")
    
    # Display the legend in an expander above the columns
    with st.expander('Legend', expanded=True):
        st.write('''
            - **Data**: [AI-Powered Job Market Insights Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset).
            - :green[Pie Chart]: Distribution of the Automation_Risk in the dataset.
            - :green[Bar Chart]: Distribution of the Job_Growth_Projection in the dataset.
            - :green[Line Chart]: Distribution of the Salary_Category in the dataset.
        ''')

    col = st.columns((3, 2, 2, 2), gap='small')

    # Automation Risk Distribution Pie Chart
    with col[0]:
            st.markdown('##### Automation Risk Distribution')
            def pie_chart_AutomationRisk():
                fig = px.pie(
            names=automationRisk_list,
            values=automationRisk_counts_list,
            title='Pie Chart of Automation Risk',
            hole=0.0, 
            height=600
                )
                st.plotly_chart(fig)
            pie_chart_AutomationRisk()
    
    with col[1]:
        
        st.markdown('###### Job Title by Automation Risk')
        scatter_plot_Automation("Job_Title", 400, 200, 1)
        
        st.markdown('###### Industry by Automation Risk')
        scatter_plot_Automation("Industry", 400, 200, 2)

        st.markdown('###### Company_Size by Automation Risk')
        scatter_plot_Automation("Company_Size", 400, 200, 3)
        
        
        
    with col[2]:
        
        st.markdown('###### Location by Automation Risk')
        scatter_plot_Automation("Location", 400, 200, 4)
        
        st.markdown('###### AI Adoption Level by Automation Risk')
        scatter_plot_Automation("AI_Adoption_Level", 400, 200, 5)
        
        st.markdown('###### Required Skills by Automation Risk')
        scatter_plot_Automation("Required_Skills", 400, 200, 6)
        
        
        
    with col[3]:   
        
        st.markdown('###### Remote Friendly by Automation Risk')
        scatter_plot_Automation("Remote_Friendly", 400, 200, 7)
        
        st.markdown('###### Job Growth Projection by Automation Risk')
        scatter_plot_Automation("Job_Growth_Projection", 400, 200, 8)
        
        st.markdown('###### Salary by Automation Risk')
        scatter_plot_Automation("Salary_USD", 400, 200, 9)
        
    st.header("💡 Insights")
    st.info("""
            
            - A fair assessment of automation risk across the dataset's different job attributes is given by the charts. An equitable distribution can be seen in the pie chart: 31.6% of jobs are medium risk, 34.6% are high risk, and 33.8% are low risk. This implies that a wide variety of employment are impacted by automation, with no specific category being impacted more than others.

            - Automation risk seems to be evenly distributed for particular job titles (such as Operations Manager, Product Manager) and locations (such as Toronto, Paris, and Tokyo), suggesting that neither factor significantly affects risk levels. Similarly, there are no significant differences in remote-friendly occupations, industry type, or company size, indicating that these variables do not significantly connect with automation risk.

            - AI adoption levels, job growth projections, and required skills like Communication, Machine Learning, and UX/UI Design also display an even distribution of automation risk, implying no specific skills or adoption levels are more susceptible to automation. Salary categories, too, are spread across all risk levels, showing that higher or lower salaries do not predict automation risk.

            - In conclusion, the data shows that automation risk is widely distributed across several job attributes, impacting a wide range of occupations, industries, and skill levels nearly equally. This wide-ranging effect emphasizes how crucial flexible approaches are as technology affects different kinds of jobs in different sectors.""")
        
    col = st.columns((3, 2, 2, 2), gap='small')
    
    with col[0]:
           
        st.markdown('##### Job Growth Distribution')
        def bar_chart_Growth():
                fig = px.bar(
        x=jobGrowth_list,  
        y=jobGrowth_counts_list,  
        title='Bar Chart of Job Growth Projection',
        labels={'x': 'Job Growth Category', 'y': 'Count'}, 
        color=jobGrowth_list,  
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=600
    )
                st.plotly_chart(fig)

        bar_chart_Growth() 
          
    
    with col[1]:
        
        st.markdown('###### Job Title by Job Growth Projection')
        scatter_plot_Growth("Job_Title", 400, 200, 1)
        
        st.markdown('###### Industry by Job Growth Projection')
        scatter_plot_Growth("Industry", 400, 200, 2)

        st.markdown('###### Company_Size by Job Growth Projection')
        scatter_plot_Growth("Company_Size", 400, 200, 3)  
        
    with col[2]:
        
        st.markdown('###### Location by Job Growth Projection')
        scatter_plot_Growth("Location", 400, 200, 4)
        
        st.markdown('###### AI Adoption Level by Job Growth Projection')
        scatter_plot_Growth("AI_Adoption_Level", 400, 200, 5)
        
        st.markdown('###### Required Skills by Job Growth Projection')
        scatter_plot_Growth("Required_Skills", 400, 200, 6) 
        
    with col[3]:   
        
        st.markdown('###### Remote Friendly by Job Growth Projection')
        scatter_plot_Growth("Remote_Friendly", 400, 200, 7)
        
        st.markdown('###### Automation Risk by Job Growth Projection')
        scatter_plot_Growth("Automation_Risk", 400, 200, 8)
        
        st.markdown('###### Salary by Job Growth Projection')
        scatter_plot_Growth("Salary_USD", 400, 200, 9)

    st.header("💡 Insights")
    st.info("""
            
            - A balanced perspective of job growth estimates across the dataset's many job qualities is given by the charts. Jobs are nearly equally likely to increase, decrease, or stay the same, according to the bar chart, suggesting a balanced outlook. All growth categories have specific job titles, such as operations manager, product manager, sales manager, and cybersecurity analyst, indicating that a job's title does not significantly influence its potential for growth.Furthermore, businesses like energy, telecommunications, education, and entertainment, as well as jobs that are conducive to remote work, exhibit an even distribution across growth, decline, and stability, indicating that these elements do not significantly affect the potential for job growth.

            - A comparable distribution is shown across all employment growth estimates for variables such as the degree of AI deployment, automation risk, company size, necessary skills, and pay levels. With abilities like communication, machine learning, JavaScript, and UX/UI design applicable across the board, this distribution suggests that these attributes have little bearing on the likelihood of job growth or decline. In general, the potential for job development is widely dispersed, indicating a diverse labor market with growth opportunities not restricted to certain positions, regions, or sectors. This emphasizes how crucial adaptability and flexibility are in a changing labor market.""")
        
    
    col = st.columns((3, 2, 2, 2), gap='small')
    
    with col[0]:
        
        st.markdown('##### Salary Category Distribution')
        def line_chart_SalaryCategory():
                fig = px.line(
                    x=salaryCategory_list, 
                    y=salaryCategory_counts_list, 
                    title=' Line Chart of Salary Category',
                    labels={'x': 'Salary Category','y': 'Count'}, 
                    markers=True, 
                    height=600
                )
                st.plotly_chart(fig)
        line_chart_SalaryCategory()  
    
    with col[1]:
        
        st.markdown('###### Job Title by Salary Category')
        scatter_plot_Salary("Job_Title", 400, 200, 1)
        
        st.markdown('###### Industry by Salary Category')
        scatter_plot_Salary("Industry", 400, 200, 2)

        st.markdown('###### Company_Size by Salary Category')
        scatter_plot_Salary("Company_Size", 400, 200, 3)  
        
    with col[2]:
        
        st.markdown('###### Location by Salary Category')
        scatter_plot_Salary("Location", 400, 200, 4)
        
        st.markdown('###### AI Adoption Level by Salary Category')
        scatter_plot_Salary("AI_Adoption_Level", 400, 200, 5)
        
        st.markdown('###### Required Skills by Salary Category')
        scatter_plot_Salary("Required_Skills", 400, 200, 6) 
        
    with col[3]:   
        
        st.markdown('###### Remote Friendly by Salary Category')
        scatter_plot_Salary("Remote_Friendly", 400, 200, 7)
        
        st.markdown('###### Automation Risk by Salary Category')
        scatter_plot_Salary("Automation_Risk", 400, 200, 8)
        
        st.markdown('###### Job Growth Projection by Salary Category')
        scatter_plot_Salary("Job_Growth_Projection", 400, 200, 9)

    st.header("💡 Insights")
    st.info("""
            
            - According to the salary category distribution, senior-level positions are the least common, while entry-level roles make up the majority of the dataset. There is room for career progression in positions with titles like operations manager, product manager, sales manager, and cybersecurity analyst, which are available at all pay levels. Similarly, entry-, mid-, and senior-level jobs are offered in places like Toronto, Paris, Tokyo, and Dubai, indicating that there are plenty of career opportunities with different pay scales available abroad. Remote work flexibility is available at various career levels, as evidenced by the fact that both remote-friendly and non-remote roles are distributed throughout all wage categories.

            - Jobs in all pay ranges can be found in industries like energy, telecommunications, education, and entertainment, indicating that advancement from entry-level to senior positions is feasible in these fields. As occupations with different levels of automation risk and AI usage offer a range of incomes, the data indicates that automation risk and AI adoption levels (high, medium, and low) do not significantly effect salary. There seem to be prospects for advancement at every income level, regardless of the size of the company—small, medium, or large. Every career level benefits from having abilities like communication, machine learning, JavaScript, and UX/UI design, which are in demand throughout entry, mid, and senior income categories.""")
        
    
# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here
    
    st.subheader("DataFrame Information")
    buffer = StringIO()  
    dataset.info(buf=buffer)  
    info = buffer.getvalue()  
    st.text(info)

    st.info("As we can see from the DataFrame information, there are no null values in any of the data types. Based on this, we can conclude that there are no data columns that need to be dropped.")

    st.subheader("Null Values of each Attributes")
    st.code("dataset.isna().sum()")
    st.write(dataset.isna().sum())
    
    st.info("There are no null values found on the dataframe.")
    
    st.subheader("Duplicate Rows")

    duplicates = dataset[dataset.duplicated(keep=False)]
    st.dataframe(duplicates)
    st.write(f"### Number of duplicate rows: {dataset.duplicated().sum()}")  
    st.info("There are no duplicate rows found on the dataframe.")
    
    st.header("Column Classification")

    col = st.columns((2, 2), gap='small')

    # Categorical columns
    with col[0]:
        cat_col = [col for col in dataset.columns if dataset[col].dtype == 'object']
        st.subheader('Categorical Columns')
        st.write(cat_col)

    # Numerical columns
    with col[1]:
        num_col = [col for col in dataset.columns if dataset[col].dtype != 'object']
        st.subheader('Numerical Columns')
        st.write(num_col)

    # Divider for better visual separation
    st.markdown("---")

    # Display unique value counts in Categorical and Numerical columns
    st.header("Unique Value Counts")

    col = st.columns((2, 2), gap='small')

    with col[0]:
        st.subheader('Categorical Columns Unique Counts')
        cat_unique_counts = dataset[cat_col].nunique().reset_index()
        cat_unique_counts.columns = ['Column', 'Unique Counts']
        st.dataframe(cat_unique_counts)

    with col[1]:
        st.subheader('Numerical Columns Unique Counts')
        num_unique_counts = dataset[num_col].nunique().reset_index()
        num_unique_counts.columns = ['Column', 'Unique Counts']
        st.dataframe(num_unique_counts)

    # Divider for separation
    st.markdown("---")

    # Display unique values for each categorical column in a grid
    st.header("Unique Values in Categorical Columns")

    # Adjust to display in a grid of five columns
    col1 = st.columns((2, 2, 2, 2, 2), gap='small')

    for i, col in enumerate(cat_col):
        with col1[i % 5]:  # Distribute items cyclically across five columns
            st.subheader(f'{col}')
            st.write(dataset[col].unique())

    # Divider for separation
    st.markdown("---")

    # Display unique values for numerical columns
    st.header("Unique Values in Numerical Columns")

    for col in num_col:
        st.subheader(f'{col}')
        st.write(dataset[col].unique())

    # Observation section for insights
    st.info("""
                - We used `df[cat_col].nunique()` to identify the number of unique values in each categorical column. 
                - The analysis shows that the dataset's categorical columns are properly initialized, confirmed using the `.unique()` method. 
                - There is only one numerical column (e.g., `Salary_USD`), which was included in the analysis.
                """)
    st.markdown("---")
    
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
    df_data['Salary_encoded'] = salary_encoder.fit_transform(df['Salary_Category'])
    df_data['Salary_USD'] = df['Salary_USD']
    """)
    
    st.subheader("Mapping of the Attributes for Prediction")
    
    col = st.columns((3, 4, 5), gap='large')

    with col[0]:
        automation_mapping_df = pd.DataFrame({
            'Original Automation Risk': dataset['Automation_Risk'],
            'Encoded Automation Risk': df_data['Automation_encoded']
            })

            # Display the DataFrame in Streamlit
        st.write(automation_mapping_df)
    
    with col[1]:
        jobGrowth_mapping_df = pd.DataFrame({
            'Original Job Growth Projection': dataset['Job_Growth_Projection'],
            'Encoded Job Growth Projection': df_data['Growth_encoded']
            })

            # Display the DataFrame in Streamlit
        st.write(jobGrowth_mapping_df)
        
    with col[2]:
        salary_mapping_df = pd.DataFrame({
            'Original Salary Category': dataset['Salary_Category'],
            'Encoded Salary Category': df_data['Salary_encoded']
            })

            # Display the DataFrame in Streamlit
        st.write(salary_mapping_df)
    
    st.markdown("---")
    
    #Train-Test split
    
    st.subheader("Train-Test Split")
    
    st.write("### For Automation Risk: ")
    
    features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Salary_USD', 'Growth_encoded']
    X1 = df_data[features]
    Y1 = df_data['Automation_encoded']
    
    st.code(""" features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Salary_USD', 'Growth_encoded']
    X1 = df_data[features]
    Y1 = df_data['Automation_encoded']""")
    
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=42)

    st.code("""X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=42)""")
    
    st.write('Training Features (X1_train):')
    st.dataframe(X1_train)

    st.write('Testing Features (X1_test):')
    st.dataframe(X1_test)

    st.write('Training Target Variable (Y1_train):')
    st.dataframe(Y1_train)

    st.write('Testing Target Variable (Y1_test):')
    st.dataframe(Y1_test)
    
    st.info("""
    *  We used `.head()` to give us a preview of the `X1_test` and `X1_train` data.
    *  We used `.shape` to give us the dimensions of the `X1_test` and `X1_train` data.
    * Same case with the Y1_train and Y1_test.

    As you can see, the test shape has a lower number than the training shape. Since we should always allot more data to training for the machine learning model to practice.
    """)
    
    clf_automation.fit(X1_train, Y1_train)
    
    y_pred = clf_automation.predict(X1_test)
    
    train_accuracy = clf_automation.score(X1_train, Y1_train) #train daTa
    test_accuracy = clf_automation.score(X1_test, Y1_test) #test daTa
    
    importance_df = pd.DataFrame({
        'Feature': X1.columns,
        'Importance': clf_automation.feature_importances_
    })

    st.session_state['importance_df'] = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    st.markdown("---")
    
    st.write("### For Growth Projection")
    
    features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Salary_USD', 'Automation_encoded']
    X2 = df_data[features]
    Y2 = df_data['Growth_encoded']
    
    st.code("""features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Salary_USD', 'Automation_encoded']
    X2 = df_data[features]
    Y2 = df_data['Growth_encoded']""")
    
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

    st.info("""
    *  We used `.head()` to give us a preview of the `X2_test` and `X2_train` data.
    *  We used `.shape` to give us the dimensions of the `X2_test` and `X2_train` data.
    * Same case with the Y2_train and Y2_test.

    As you can see, the test shape has a lower number than the training shape. Since we should always allot more data to training for the machine learning model to practice.
    """)
    
    clf_growthPrediction.fit(X2_train, Y2_train)
    
    y_pred = clf_growthPrediction.predict(X2_test)
    
    train_accuracy = clf_growthPrediction.score(X2_train, Y2_train) #train daTa
    test_accuracy = clf_growthPrediction.score(X2_test, Y2_test) #test daTa
    
    importance_df_Growth = pd.DataFrame({
        'Feature': X2.columns,
        'Importance_Growth': clf_growthPrediction.feature_importances_
    })

    st.session_state['importance_df_Growth'] = importance_df_Growth.sort_values(by='Importance_Growth', ascending=False).reset_index(drop=True)
     
    st.markdown("---")
    
    st.write("### For Salary Category")
    
    features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Automation_encoded', 'Growth_encoded']
    X4 = df_data[features]
    Y4 = df_data['Salary_encoded']
    
    st.code("""features = ['Job_encoded', 'Industry_encoded', 'Size_encoded', 'Location_encoded', 'AI_Adoption_encoded','Skills_encoded', 'Remote_encoded', 'Automation_encoded', 'Growth_encoded']
    X4 = df_data[features]
    Y4 = df_data['Salary_encoded']""")
    
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
    
    clf_salary.fit(X4_train, Y4_train)
    
    y_pred = clf_salary.predict(X4_test)
    
    train_accuracy = clf_salary.score(X4_train, Y4_train) #train daTa
    test_accuracy = clf_salary.score(X4_test, Y4_test) #test daTa
    
    
    importance_df_Salary = pd.DataFrame({
        'Feature': X4.columns,
        'Importance_Salary': clf_salary.feature_importances_
    })

    st.session_state['importance_df_Salary'] = importance_df_Salary.sort_values(by='Importance_Salary', ascending=False).reset_index(drop=True)

    new_df = dataset

    st.write(new_df.head())

    salaryCategory_counts = new_df['Salary_Category'].value_counts()
    st.write(salaryCategory_counts)

    # Assuming your DataFrame is named 'weather_df_new'
    selected_salaryCategory = ["Mid Level", "Senior Level"]

    # Filter the DataFrame to keep only the desired summaries
    new_df_filtered = new_df[new_df['Salary_Category'].isin(selected_salaryCategory)]

    # Reset the index if needed
    new_df_filtered = new_df_filtered.reset_index(drop=True)

    st.write(new_df_filtered.head())

    salaryCategory_counts = new_df_filtered['Salary_Category'].value_counts()
    st.write(salaryCategory_counts)

    # Initialize an empty dataframe to store balanced data
    balanced_new_df = pd.DataFrame()

    # Loop through each category and sample 6547 rows (for Overcast, we'll use all rows)
for Salary_Category_Projection in salaryCategory_counts.index:
    if Salary_Category_Projection == 'Entry Level':
        sampled_df = new_df_filtered[new_df_filtered['Salary_Category'] == Salary_Category_Projection]
    else:
        sampled_df = new_df_filtered[new_df_filtered['Salary_Category'] == Salary_Category_Projection].sample(172, random_state=42)

    # Append the sampled data to the balanced dataframe
    balanced_new_df = pd.concat([balanced_new_df, sampled_df])

    Reset index if necessary
    balanced_new_df.reset_index(drop=True, inplace=True)

    # Now, 'balanced_weather_df' contains the balanced rows
    st.write(balanced_new_df['Salary_Category'].value_counts())

    st.code( """ balanced_new_df['Salary_encoded'] = growth_encoder.fit_transform(balanced_new_df['Salary_Category']) """)
    balanced_new_df['Salary_encoded'] = growth_encoder.fit_transform(balanced_new_df['Salary_Category'])
    st.write(balanced_new_df.head())
    
    st.write(balanced_new_df['Salary_Category'].unique())

    st.write(balanced_new_df['Salary_encoded'].unique())

    st.code("""
    # Mapping of the Summary and their encoded equivalent
    balanced_unique_salaryCategory = balanced_new_df['Salary_Category'].unique()
    balanced_unique_salaryCategory_encoded = balanced_new_df['Salary_encoded'].unique()
    
    # Create a new DataFrame
    balanced_salaryCategory_mapping_df = pd.DataFrame({'Summary': balanced_unique_salaryCategory, 'Summary_Encoded': balanced_unique_salaryCategory_encoded})
    """)

    balanced_unique_salaryCategory = balanced_new_df['Salary_Category'].unique()
    balanced_unique_salaryCategory_encoded = balanced_new_df['Salary_encoded'].unique()

    balanced_salaryCategory_mapping_df = pd.DataFrame({'Summary': balanced_unique_salaryCategory, 'Summary_Encoded': balanced_unique_salaryCategory_encoded})
    # Display the DataFrame
    st.write(balanced_salaryCategory_mapping_df)

    balanced_new_df['Job_encoded'] = job_encoder.fit_transform(balanced_new_df['Job_Title'])
    balanced_new_df['Industry_encoded'] = industry_encoder.fit_transform(balanced_new_df['Industry'])
    balanced_new_df['Location_encoded'] = location_encoder.fit_transform(balanced_new_df['Location'])
    balanced_new_df['Skills_encoded'] = skills_encoder.fit_transform(balanced_new_df['Required_Skills'])

    # Select features and target variable
    features = ['Job_encoded', 'Industry_encoded', 'Location_encoded', 'Skills_encoded']
    X5 = balanced_new_df[features]
    Y5 = balanced_new_df['Salary_encoded']

    st.code("""features = ['Job_encoded', 'Industry_encoded', 'Location_encoded', 'Skills_encoded']
    X5 = df_data[features]
    Y5 = df_data['Salary_encoded']""")
    
    X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5, Y5, test_size=0.3, random_state=42)
    st.code("""X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5, Y5, test_size=0.3, random_state=42)""")

    st.write('Training Features (X5_train):')
    st.dataframe(X5_train)

    st.write('Testing Features (X5_test):')
    st.dataframe(X5_test)

    st.write('Training Target Variable (Y5_train):')
    st.dataframe(Y5_train)

    st.write('Testing Target Variable (Y5_test):')
    st.dataframe(Y5_test)
    
    clf_salary.fit(X5_train, Y5_train)
    
    y_pred = clf_salary.predict(X5_test)
    
    train_accuracy = clf_salary.score(X5_train, Y5_train) #train daTa
    test_accuracy = clf_salary.score(X5_test, Y5_test) #test daTa
    st.write(train_accuracy)
    st.write(test_accuracy)
    
    importance_df_Salary = pd.DataFrame({
        'Feature': X5.columns,
        'Importance_Salary': clf_salary.feature_importances_
    })

    st.session_state['importance_df_Salary'] = importance_df_Salary.sort_values(by='Importance_Salary', ascending=False).reset_index(drop=True)

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here
    st.subheader("Random Tree Classifier")
    st.markdown("""

    Mostly used for classification tasks, the Scikit-learn library's **Decision Tree Classifier** is one of many machine learning methods. The main aim of this method is Data Point Classification. This will be is accomplished by dividing the data into ever smaller subsets according to questions. This will result in a "Tree" structure, where each of the node in the tree stands for a question or a decision point based on the data feature. Depending on the response, the data travels down a branch of the tree to a new node that poses a new query or choice.

    `Reference:` https://miro.medium.com/v2/resize:fit:1100/format:webp/1*i69vGs4AfhdhDUOlaPVLSA.png         
                
    """)
    
      # Columns to center the Decision Tree Parts image
    col_dt_fig = st.columns((2, 4, 2), gap='medium')

    with col_dt_fig[0]:
        st.write(' ')

    with col_dt_fig[1]:
        decision_tree_parts_image = Image.open('assets/RFC.png')
        st.image(decision_tree_parts_image, caption='Random Forest Classifier parts')

    with col_dt_fig[2]:
        st.write(' ')
        
    st.subheader("Training the Random Forest Classifier model for Automation Risk")
    
    st.code("""
    * clf_automation = joblib.load('models/RFC_Automation.joblib')
    * clf_automation.fit(X1_train, Y1_train)
    """)
    
    st.subheader("Model Evaluation")
    
    st.code("""

    train_accuracy = clf_automation.score(X1_train, Y1_train) #train daTa
    test_accuracy = clf_automation.score(X1_test, Y1_test) #test daTa

    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
            
    """)
    st.write("""

    **Train Accuracy:** 100.00%\n
    **Test Accuracy:** 43.33%      
             
    """)
    
    st.subheader("Feature Importance")
    
    st.code("""

    feature_importance = clf.feature_importances_
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    """)
    
    rfc_feature_importance_df = st.session_state['importance_df']

    st.dataframe(rfc_feature_importance_df, use_container_width=True, hide_index=True)
    
    def feature_importance_plot1(feature_importance_df, width=500, height=500, key='default'):
    # Generate a bar plot for feature importances
        feature_importance_fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
            orientation='h'  
        )

        feature_importance_fig.update_layout(
            width=width,  
            height=height  
        )

        # Display the plot in Streamlit
        st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot1_{key}")

    # Example DataFrame to illustrate (replace this with your actual importance DataFrame)
    rfc_feature_importance_df = pd.DataFrame({
         'Feature': [
        'Salary_USD',
        'Industry_encoded',
        'Location_encoded',
        'Job_encoded',
        'Skills_encoded',
        'AI_Adoption_encoded',
        'Growth_encoded',
        'Size_encoded',
        'Remote_encoded'
    ],
    'Importance': [
        0.22352831025114384,  # Salary_USD
        0.14118903524561172,  # Industry_encoded
        0.13812936063719536,  # Location_encoded
        0.13717709996178903,  # Job_encoded
        0.13068020124713864,  # Skills_encoded
        0.06745537056820566,  # AI_Adoption_encoded
        0.06482367698936438,  # Growth_encoded
        0.06010086796585386,  # Size_encoded
        0.03691607713369761    # Remote_encoded
    ]
    })

    # Call the function with appropriate arguments
    feature_importance_plot1(rfc_feature_importance_df, width=500, height=500, key='2')
    
    st.info("""
    Upon running . `feature_importances` in the `Random Forest Classifier Model` to check how each Growth Prediction's features influence the training of our model, it is clear that Salary_USD holds the most influence in our model's decisions having 0.2235 or 22% importance. This is followed by Location_encoded, and Industry_encoded which is closely behind of Salary_USD having 0.17 or 17% importance followed closely by Skills_encoded with 0.14 or 14%.
    """)
    
    st.markdown("---")
    
    st.subheader("Training the Random Forest Classifier model for Growth Prediction")
    
    st.code("""
    * clf_growthPrediction = joblib.load('models/RFC_GrowthPrediction.joblib')
    * clf_growthPrediction.fit(X2_train, Y2_train)
    """)
    
    st.subheader("Model Evaluation")
    
    st.code("""

    train_accuracy = clf_growthPrediction.score(X2_train, Y2_train) #train daTa
    test_accuracy = clf_growthPrediction.score(X2_test, Y2_test) #test daTa

    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
            
    """)
    st.write("""

    **Train Accuracy:** 100.00%\n
    **Test Accuracy:** 42.00%      
             
    """)
    
    st.subheader("Feature Importance")
    
    st.code("""

    feature_importance = clf_growthPrediction.feature_importances_
    st.session_state['importance_df_Growth'] = importance_df_Growth.sort_values(by='Importance_Growth', ascending=False).reset_index(drop=True)
    
    """)
    
    rfc_feature_importance_df = st.session_state['importance_df_Growth']

    st.dataframe(rfc_feature_importance_df, use_container_width=True, hide_index=True)
    
    def feature_importance_plot2(feature_importance_df, width=500, height=500, key='default'):
    # Generate a bar plot for feature importances
        feature_importance_fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
            orientation='h'  # Horizontal bar plot
        )

        # Adjust the height and width
        feature_importance_fig.update_layout(
            width=width,  # Set the width
            height=height  # Set the height
        )

        # Display the plot in Streamlit
        st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot2_{key}")

    # Example DataFrame to illustrate (replace this with your actual importance DataFrame)
    rfc_feature_importance_df = pd.DataFrame({
        'Feature': [
        'Salary_USD',
        'Skills_encoded',
        'Industry_encoded',
        'Job_encoded',
        'Location_encoded',
        'AI_Adoption_encoded',
        'Automation_encoded',
        'Size_encoded',
        'Remote_encoded'
    ],
    'Importance': [
        0.22528175689171243,  # Salary_USD
        0.13880741712136976,  # Skills_encoded
        0.1364568017263118,   # Industry_encoded
        0.13600111833404468,  # Job_encoded
        0.1322094777615761,   # Location_encoded
        0.06566605173639027,  # AI_Adoption_encoded
        0.06424098181751703,  # Automation_encoded
        0.0613686496097606,   # Size_encoded
        0.03996774500131726    # Remote_encoded
    ]
    })

    # Call the function with appropriate arguments
    feature_importance_plot2(rfc_feature_importance_df, width=500, height=500, key='2')
    
    st.info("""
    Upon running . `feature_importances` in the `Random Forest Classifier Model` to check how each Salary Category's features influence the training of our model, it is clear that Salary_USD holds the most influence in our model's decisions having 0.2253 or 22% importance. This is followed by Location_encoded, and Skills_encoded which is closely behind of Salary_USD having 0.1388 or 13% importance followed closely by Skills_encoded with 0.14 or 14%.
    """)
    
    print(f"Number of trees made: {len(clf_automation.estimators_)}")

    st.subheader("Number of Trees")
    st.code("""

    print(f"Number of trees made: {len(clf_automation.estimators_)}")
     
    """)

    st.markdown("**Number of trees made:** 100")
    
    st.subheader("Plotting the Forest")
    
    
    
    st.markdown("---")
    
    st.subheader("Training the Random Forest Classifier model for Salary Category")
    
    st.code("""
    * clf_salary = joblib.load('models/RFC_Salary.joblib')
    * clf_salary.fit(X4_train, Y4_train)
    """)
    
    st.subheader("Model Evaluation")
    
    st.code("""

    train_accuracy = clf_salary.score(X4_train, Y4_train) #train daTa
    test_accuracy = clf_salary.score(X4_test, Y4_test) #test daTa

    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
            
    """)
    st.write("""

    **Train Accuracy:** 100.00%\n
    **Test Accuracy:** 58.00%      
             
    """)
    
    st.subheader("Feature Importance")
    
    st.code("""

    feature_importance = clf_salary.feature_importances_
    st.session_state['importance_df_Salary'] = importance_df_Salary.sort_values(by='Importance_Salary', ascending=False).reset_index(drop=True)
    
    """)
    
    rfc_feature_importance_df = st.session_state['importance_df_Salary']

    st.dataframe(rfc_feature_importance_df, use_container_width=True, hide_index=True)
    
    def feature_importance_plot3(feature_importance_df, width=500, height=500, key='default'):
    # Generate a bar plot for feature importances
        feature_importance_fig = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
            orientation='h'  # Horizontal bar plot
        )

        # Adjust the height and width
        feature_importance_fig.update_layout(
            width=width,  # Set the width
            height=height  # Set the height
        )

        # Display the plot in Streamlit
        st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot3_{key}")

    # Example DataFrame to illustrate (replace this with your actual importance DataFrame)
    rfc_feature_importance_df = pd.DataFrame({
          'Feature': [
        'Skills_encoded',
        'Job_encoded',
        'Industry_encoded',
        'Location_encoded',
        'Size_encoded',
        'Growth_encoded',
        'AI_Adoption_encoded',
        'Automation_encoded',
        'Remote_encoded'
    ],
    'Importance': [
        0.1721702260381549,  # Skills_encoded
        0.1600515981232281,  # Job_encoded
        0.15794542574315804, # Industry_encoded
        0.1551749173546094,  # Location_encoded
        0.0812005326432557,  # Size_encoded
        0.07959563749630556, # Growth_encoded
        0.07460182528071888, # AI_Adoption_encoded
        0.07072907265913353, # Automation_encoded
        0.048530764661435785 # Remote_encoded
    ]
    })

    # Call the function with appropriate arguments
    feature_importance_plot3(rfc_feature_importance_df, width=500, height=500, key='2')
    
    st.info("""
    Upon running . `feature_importances` in the `Random Forest Classifier Model` to check how each Automation_Risk's features influence the training of our model, it is clear that Salary_USD holds the most influence in our model's decisions having 0.2253 or 22% importance. This is followed by Location_encoded, and Skills_encoded which is closely behind of Salary_USD having 0.1388 or 13% importance followed closely by Skills_encoded with 0.14 or 14%.
    """)


# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.markdown("<h1 style='text-align: center;'>🎲 Random Forest Classifier</h1>", unsafe_allow_html=True)

    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here
    col_pred = st.columns((1, 1, 1, 1), gap='medium')
    
    # Initialize session state for clearing results
    if 'clear' not in st.session_state:
        st.session_state.clear = False
    
    
    with col_pred[0]:
        with st.expander('Options', expanded=True):
            show_dataset = st.checkbox('Show Dataset')
            show_classes = st.checkbox('Show All Prediction Classes')
            
            st.write("#### Automation Risk")
            show_HighAutomationRisk = st.checkbox('Show High')
            show_MediumAutomationRisk = st.checkbox('Show Medium')
            show_LowAutomationRisk = st.checkbox('Show Low')
            
            st.write("#### Growth Prediction")
            show_Growth = st.checkbox('Show Growth')
            show_Decline = st.checkbox('Show Decline')
            show_Stable = st.checkbox('Show Stable')
            
            st.write("#### Salary Category")
            show_Entry = st.checkbox('Show Entry Level')
            show_Mid = st.checkbox('Show Mid Level')
            show_Senior = st.checkbox('Show Senior Level')

            clear_results = st.button('Clear Results', key='clear_results')

            if clear_results:
                st.session_state.clear = True

    
    with col_pred[1]:
       
        
        # Input boxes for the features
        
        #JobTitle
        job_classes_list = df_data['Job_encoded'].unique()  # Get unique encoded values
        job_titles = dataset['Job_Title'].unique()  # Assuming this has the original job titles
        job_mapping = dict(zip(job_titles, job_classes_list))
        selected_job = st.radio('Select Job Title', options=job_titles)
        job_encoded_value = job_mapping[selected_job]
        #Industry
        industry_classes_list = df_data['Industry_encoded'].unique()  # Get unique encoded values
        industry = dataset['Industry'].unique()  # Assuming this has the original job titles
        industry_mapping = dict(zip(industry, industry_classes_list))
        selected_industry = st.radio('Select Industry', options=industry)
        industry_encoded_value = industry_mapping[selected_industry]
    with col_pred[2]:
        #Size
        size_classes_list = df_data['Size_encoded'].unique()  # Get unique encoded values
        companySize = dataset['Company_Size'].unique()  # Assuming this has the original job titles
        size_mapping = dict(zip(companySize, size_classes_list))
        selected_size = st.radio('Select Company Size', options=companySize)
        size_encoded_value = size_mapping[selected_size]
        #Location
        location_classes_list = df_data['Location_encoded'].unique()  # Get unique encoded values
        location = dataset['Location'].unique()  # Assuming this has the original job titles
        location_mapping = dict(zip(location, location_classes_list))
        selected_location = st.radio('Select Location', options=location)
        location_encoded_value = location_mapping[selected_location]
        #AI_Adoption
        aiAdoption_classes_list = df_data['AI_Adoption_encoded'].unique()  # Get unique encoded values
        aiAdoption = dataset['AI_Adoption_Level'].unique()  # Assuming this has the original job titles
        aiAdoption_mapping = dict(zip(aiAdoption, aiAdoption_classes_list))
        selected_aiAdoption = st.radio('Select AI Adoption', options=aiAdoption)
        AiAdoption_encoded_value = aiAdoption_mapping[selected_aiAdoption]
    with col_pred[3]:
        #Skills
        skills_classes_list = df_data['Skills_encoded'].unique()  # Get unique encoded values
        skills = dataset['Required_Skills'].unique()  # Assuming this has the original job titles
        skills_mapping = dict(zip(skills, skills_classes_list))
        selected_skills = st.radio('Select Skills', options=skills)
        skills_encoded_value = skills_mapping[selected_skills]
        #Remote
        remote_classes_list = df_data['Remote_encoded'].unique()  # Get unique encoded values
        remote = dataset['Remote_Friendly'].unique()  # Assuming this has the original job titles
        remote_mapping = dict(zip(remote, remote_classes_list))
        selected_remote = st.radio('Remote Friendly?', options=remote)
        remote_encoded_value = remote_mapping[selected_remote]
        #Growth
        growth_classes_list = df_data['Growth_encoded'].unique()  # Get unique encoded values
        growth = dataset['Job_Growth_Projection'].unique()  # Assuming this has the original job titles
        growth_mapping = dict(zip(growth, growth_classes_list))
        selected_growth = st.radio('Select Growth Projection', options=growth)
        growth_encoded_value = growth_mapping[selected_growth]
        #AutomationRisk
        automationRisk_classes_list = df_data['Automation_encoded'].unique()  # Get unique encoded values
        automationRisk = dataset['Automation_Risk'].unique()  # Assuming this has the original job titles
        automationRisk_mapping = dict(zip(automationRisk, automationRisk_classes_list))
        selected_automationRisk = st.radio('Select Automation Risk', options=automationRisk)
        automationRisk_encoded_value = automationRisk_mapping[selected_automationRisk]
        
        #Salary
        dt_SalaryUSD = st.number_input('Input Salary USD', min_value=0.0, max_value=5000000.0, step=10000.00, key='dt_SalaryUSD', value=0.0 if st.session_state.clear else st.session_state.get('dt_SalaryUSD', 0.0))
    
    with col_pred[0]:    
        #Automation Risk Detection
        
        automation_classes_list = ['High', 'Low', 'Medium']
        
        # Button to detect the Automation Risk
        if st.button('Detect Automation Risk', key='dt_detectAutomation'):
            # Prepare the input data for prediction
            dt_input_data = [[job_encoded_value, industry_encoded_value, size_encoded_value, location_encoded_value, AiAdoption_encoded_value, skills_encoded_value, remote_encoded_value, dt_SalaryUSD, growth_encoded_value]] 
            
            # Predict the Iris species
            dt_prediction = clf_automation.predict(dt_input_data)
            
            # Display the prediction result
            st.markdown(f'The predicted Automation Risk is: `{automation_classes_list[dt_prediction[0]]}`')
            
        #Growth Projection Detection

        growth_classes_list = ['Decline', 'Growth', 'Stable']
        
        # Button to detect the Growth Projection
        if st.button('Detect Growth Projection', key='dt_detectGrowth'):
            # Prepare the input data for prediction
            dt_input_data = [[job_encoded_value, industry_encoded_value, size_encoded_value, location_encoded_value, AiAdoption_encoded_value, skills_encoded_value, remote_encoded_value, dt_SalaryUSD, automationRisk_encoded_value]] 
            
            # Predict the Iris species
            dt_prediction = clf_growthPrediction.predict(dt_input_data)
            
            # Display the prediction result
            st.markdown(f'The predicted Growth Projection is: `{growth_classes_list[dt_prediction[0]]}`')
            
        #Salary Category Detection

        salary_classes_list = ['Entry Level', 'Mid Level', 'Senior Level']
        
        # Button to detect the Salary Category
        if st.button('Detect Salary Category', key='dt_salaryCategory'):
            # Prepare the input data for prediction
            dt_input_data = [[job_encoded_value, industry_encoded_value, size_encoded_value, location_encoded_value, AiAdoption_encoded_value, skills_encoded_value, remote_encoded_value, automationRisk_encoded_value, growth_encoded_value]] 
            
            # Predict the Iris species
            dt_prediction = clf_salary.predict(dt_input_data)
            
            # Display the prediction result
            st.markdown(f'The predicted Salary Category is: `{salary_classes_list[dt_prediction[0]]}`')

    # Create 3 Data Frames containing  5 rows for each 
    high_samples = dataset[dataset["Automation_Risk"] == "High"]
    medium_samples = dataset[dataset["Automation_Risk"] == "Medium"]
    low_samples = dataset[dataset["Automation_Risk"] == "Low"]
    
    growth_samples = dataset[dataset["Job_Growth_Projection"] == "Growth"]
    stable_samples = dataset[dataset["Job_Growth_Projection"] == "Stable"]
    decline_samples = dataset[dataset["Job_Growth_Projection"] == "Decline"]
    
    entry_samples = dataset[dataset["Salary_Category"] == "Entry Level"]
    mid_samples = dataset[dataset["Salary_Category"] == "Mid Level"]
    senior_samples = dataset[dataset["Salary_Category"] == "Senior Level"]

    if show_dataset:
        # Display the dataset
        st.subheader("Dataset")
        st.dataframe(dataset, use_container_width=True, hide_index=True)

    if show_classes:
        st.subheader("High Automation Risk")
        st.dataframe(high_samples.head(5), use_container_width=True, hide_index=True)
        st.subheader("Medium Automation Risk")
        st.dataframe(medium_samples.head(5), use_container_width=True, hide_index=True)
        st.subheader("Low Automation Risk")
        st.dataframe(low_samples.head(5), use_container_width=True, hide_index=True)
        st.subheader("Growth Projection: Growth")
        st.dataframe(growth_samples.head(5), use_container_width=True, hide_index=True)
        st.subheader("Growth Projection: Stable")
        st.dataframe(stable_samples.head(5), use_container_width=True, hide_index=True)
        st.subheader("Growth Projection: Decline")
        st.dataframe(decline_samples.head(5), use_container_width=True, hide_index=True)
        st.subheader("Entry Level Salary")
        st.dataframe(entry_samples.head(5), use_container_width=True, hide_index=True)
        st.subheader("Mid Level Salary")
        st.dataframe(mid_samples.head(5), use_container_width=True, hide_index=True) 
        st.subheader("Senior Level Salary")
        st.dataframe(senior_samples.head(5), use_container_width=True, hide_index=True)
        
    #Automation Risk

    if show_HighAutomationRisk:
        st.subheader("High Automation Risk")
        st.dataframe(high_samples, use_container_width=True, hide_index=True)
    
    if show_MediumAutomationRisk:
        st.subheader("Medium Automation Risk")
        st.dataframe(medium_samples, use_container_width=True, hide_index=True)
        
    if show_LowAutomationRisk:
        st.subheader("Low Automation Risk")
        st.dataframe(low_samples, use_container_width=True, hide_index=True)
        
    #JobGrowth Projection

    if show_Growth:
        st.subheader("Growth Projection: Growth")
        st.dataframe(growth_samples, use_container_width=True, hide_index=True)
    
    if show_Stable:
        st.subheader("Growth Projection: Stable")
        st.dataframe(stable_samples, use_container_width=True, hide_index=True)
        
    if show_Decline:
        st.subheader("Growth Projection: Decline")
        st.dataframe(decline_samples, use_container_width=True, hide_index=True)
        
    #Salary Category

    if show_Entry:
        st.subheader("Entry Level Salary")
        st.dataframe(entry_samples, use_container_width=True, hide_index=True)
    
    if show_Mid:
        st.subheader("Mid Level Salary")
        st.dataframe(mid_samples, use_container_width=True, hide_index=True)
        
    if show_Senior:
        st.subheader("Senior Level Salary")
        st.dataframe(senior_samples, use_container_width=True, hide_index=True)
        
        
    
        
# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
    st.markdown("""
                
    Through exploratory data analysis and training of the classification model (`Random Forest Classifier`) on the **AI-Powered Job Market Insights dataset** , the key insights and observations are:

    #### 1. 📊 **Dataset Characteristics**:
    The dataset, which consists of 500 rows, includes important information such as Job_Growth_Projection, Automation_Risk, and Salary_USD. The balanced distribution of job types and qualities ensures a comprehensive knowledge of AI's impact on employment patterns without requiring extensive data rebalancing.

    #### 2. 📝 **Feature Distributions and Separability**:
    Different distributions for Job_Growth_Projection, Automation_Risk, and Salary_USD were found using an exploratory analysis. Higher salary in particular industries and company sizes, as well as variations in automation risk among AI-adopting firms, are some of the trends noted. AI_Adoption_Level's distribution across job categories demonstrates how adoption levels affect automation risk, particularly for tasks that are heavily automated.

    #### 3. ⚙️ **Random Forest Classifier (Automation Risk Prediction)**:
    The Random Forest Classifier model trained on Automation_Risk achieved high accuracy on training data (100%) but showed a notable performance drop (43.33%) on test data. Feature importance analysis indicated *Salary_USD* as the most influential feature, contributing 22%, followed by *Location_encoded* and *Skills_encoded* at approximately 13% each, reflecting the impact of salary and job location on automation risk.
    
    #### 4. 🚀 **Random Forest Classifier (Growth Prediction)**:
    For Growth Prediction, the Random Forest Classifier displayed similar results, with 100% accuracy on training data and 42% on test data. Here, *Salary_USD* again emerged as the most significant predictor at 22%, followed by *Location_encoded* (17%) and *Industry_encoded* (14%), suggesting that job growth is influenced by salary and industry context.
    
    #### 5. 💸 **Random Forest Classifier (Salary Category Prediction)**:
    The Salary Category model achieved 100% accuracy on training data and 58% on test data. In this model, *Salary_USD* was the dominant feature, contributing 22% to prediction accuracy, with *Location_encoded* and *Skills_encoded* also highly influential, reinforcing the relationship between salary, location, and required skills in classification tasks.
    
    #### **Summing up:**  
    This experiment demonstrated the importance of feature-rich, balanced datasets, such as the AI-Powered Job Market Insights dataset, for comprehending the growth trends and automation risk of different professions. Using both Random Forest and Decision Tree Classifiers, we found that variables like *Salary_USD*, *Location*, and *Industry* were essential for predicting the risk and expansion of job automation. Strong model learning is shown by high training accuracy, but overfitting is suggested by the performance difference on test data, indicating that additional fine-tuning or complexity changes are needed for subsequent applications. The basis for evaluating AI's effects on job security and industry growth prospects is provided by this analysis.
    """)
