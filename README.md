
# Automation Risk, Job Growth Projection, Salary Category Dashboard using Streamlit

This is a Streamlit web application that performs Exploratory Data Analysis (EDA), Data Preprocessing, and Supervised Machine Learning in order for this to work the machine utilizes Decision Tree Classifier and Random Forest Classifier to classify classify the Automation Risk, Job Growth Projection, Salary Category.

![Main Page Screenshot](screenshots/IrisClassificationDashboard.webp)

### 🔗 Links:

- 🌐 [Streamlit Link](https://abnederio-datasciproj-dashmain-wnmtxh.streamlit.app/)
- 📗 [Google Colab Notebook](https://colab.research.google.com/drive/1ZdLgzKjkpj3I7VR9rSFYE653MVEZ6Z1u#scrollTo=LqM83prKuuEF)

### 📊 Dataset:

- [AI-Powered Job Market Insights (Kaggle)](https://www.kaggle.com/datasets/uom190346a/ai-powered-job-market-insights)

### 📖 Pages:

1. Dataset - Brief description of the AI-Powered Job Market Insights dataset used in this dashboard.
2.EDA - Exploratory Data Analysis of the AI-Powered Job Market Insights dataset, highlighting the distribution of 3.Salary_USD, Automation_Risk, and Job_Growth_Projection. This section includes graphs such as Pie Charts and Bar 4.Charts.
5.Data Cleaning / Pre-processing - Overview of data cleaning and pre-processing steps, including encoding for 6.Salary_USD, Automation_Risk, and Job_Growth_Projection columns, as well as splitting the dataset into training and testing sets.
7.Machine Learning - Training a supervised classification model using the Random Forest Classifier. This section covers model evaluation, feature importance, and tree plotting.
8.Prediction - A prediction page where users can input values to predict Salary, Automation Risk, and Job Growth 9.Projection using the trained models.
Conclusion - A summary of the insights and observations from the EDA and model training.

### 💡 Findings / Insights

With the use of exploratory data analysis and training the classification models (`Decision Tree Classifier` and `Random Forest Regressor`) on the AI-Powered Job Market Insights, the groups observations are:
	According to the statistics, the distribution of salaries, job growth potential, and automation danger is uniform across a range of job variables, including job titles, locations, industries, and firm sizes. Jobs at high, medium, and low levels are all nearly equally affected by automation risk; there are no appreciable distinctions between job categories or locations. A balanced distribution of automation risk may also be seen in AI adoption, skill requirements, and pay ranges, indicating that automation's effects are widespread and not limited to certain industries or job roles.

	The potential for employment growth also seems to be fairly distributed across many industries and occupations, suggesting that growth, stability, and decline are not unique to certain job titles, geographical areas, or skill sets. Although entry-level jobs are the most prevalent, independent of automation risk or AI adoption, opportunities for growth exist across all industries and wage levels. Across all career levels, in-demand talents like communication, machine learning, JavaScript, and UX/UI design are recognized, highlighting the importance of flexibility and a diverse skill set in navigating a technologically driven labor market.

#### 1. 📊 **Dataset Characteristics**:

	The dataset, which consists of 500 rows, includes important information such as Job_Growth_Projection, Automation_Risk, and Salary_USD. The balanced distribution of job types and qualities ensures a comprehensive knowledge of AI's impact on employment patterns without requiring extensive data rebalancing.

#### 2. 📝 **Feature Distributions and Separability**:

	Different distributions for Job_Growth_Projection, Automation_Risk, and Salary_USD were found using an exploratory analysis. Higher salary in particular industries and company sizes, as well as variations in automation risk among AI-adopting firms, are some of the trends noted. AI_Adoption_Level's distribution across job categories demonstrates how adoption levels affect automation risk, particularly for tasks that are heavily automated.

#### 3. 📈 **Model Performance (Decision Tree Classifier)**:

	The Random Forest Classifier model trained on Automation_Risk achieved high accuracy on training data (100%) but showed a notable performance drop (43.33%) on test data. Feature importance analysis indicated Salary_USD as the most influential feature, contributing 22%, followed by Location_encoded and Skills_encoded at approximately 13% each, reflecting the impact of salary and job location on automation risk.
#### 4. 📈 **Model Performance (Random Forest Regressor)**:

	For Growth Prediction, the Random Forest Classifier displayed similar results, with 100% accuracy on training data and 42% on test data. Here, Salary_USD again emerged as the most significant predictor at 22%, followed by Location_encoded (17%) and Industry_encoded (14%), suggesting that job growth is influenced by salary and industry context.

	The Salary Category model achieved 100% accuracy on training data and 58% on test data. In this model, Salary_USD was the dominant feature, contributing 22% to prediction accuracy, with Location_encoded and Skills_encoded also highly influential, reinforcing the relationship between salary, location, and required skills in classification tasks.
##### **Summing up:**

	This experiment demonstrated the importance of feature-rich, balanced datasets, such as the AI-Powered Job Market Insights dataset, for comprehending the growth trends and automation risk of different professions. Using both Random Forest and Decision Tree Classifiers, we found that variables like Salary_USD, Location, and Industry were essential for predicting the risk and expansion of job automation. Strong model learning is shown by high training accuracy, but overfitting is suggested by the performance difference on test data, indicating that additional fine-tuning or complexity changes are needed for subsequent applications. The basis for evaluating AI's effects on job security and industry growth prospects is provided by this analysis.

