import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Kidney Disease Prediction Dashboard", layout="wide")

st.title("Kidney Disease Prediction Dashboard")

#Load the dataset
data_kidney =pd.read_csv('kidney_disease_dataset.csv')

st.subheader("Kidney Disease Dataset Overview")

st.write("This dataset contains information about patients with kidney disease. The features include various medical attributes and the target variable indicates whether the patient has kidney disease.")

#Displaying the  dataset
st.subheader("Dataset Preview")
overview = st.checkbox("Show dataset overview")
if overview:
    st.write("Here is a preview of the dataset:")
    st.dataframe(data_kidney)

#display the shape of the dataset
st.subheader("Dataset Shape")
st.write(f"The dataset contains {data_kidney.shape[0]} rows and {data_kidney.shape[1]} columns.")

# Displaying the columns of the dataset
st.subheader("Dataset Columns")
st.write("The dataset contains the following columns:")
st.write(data_kidney.columns.tolist())

#display the datatypes of the columns
st.subheader("Data Types of Columns")
st.write("The data types of the columns are as follows:")
st.write(data_kidney.dtypes)

# Displaying the summary statistics of the dataset
st.subheader("Summary Statistics")
summary_stats = st.checkbox("Show summary statistics")
if summary_stats:
    st.write("Here are the summary statistics of the dataset:")
    st.dataframe(data_kidney.describe())

data_analysis = data_kidney.copy()
data_analysis.replace({
    'CKD_Status': {1: 'CKD', 0: 'Not CKD'},
    'Dialysis_Needed': {1: 'Yes', 0: 'No'},
    'Diabetes': {1: 'Yes', 0: 'No'},
    'Hypertension': {1: 'Yes', 0: 'No'}
}, inplace=True)

st.subheader("Data Analysis")
st.write("This section will provide insights innto the dataset through various visualizations and analysis")

#show head of the mapped dataset
st.markdown("**Review of dataset after mapping categorical variables**")
st.dataframe(data_analysis.head())


custom_palette = ["#1B2558", "#856506", "#0b398a", "#f8b71f", "#0D048F", "#bb931d", "#0B053A",
                  "#cf9202"]

blue_palette = ["#358ff7", "#125ee0", "#0E0681", "#0B053A"]
orange_palette = ["#856506", "#bb931d", "#e0a009", "#e6b951"]
#statistical analysis 
st.markdown("**Statistical Analysis**")

target_options = st.multiselect("'Select Target Variable for Analysis'",["CKD_Status", "Dialysis_Needed"], default=None)
st.write("Selected Target Variable(s):", target_options)
if target_options:
    for target in target_options:
        st.subheader(f"Statistical Analysis for {target}")
        st.write(data_analysis[target].value_counts())
        st.write(data_analysis[target].value_counts(normalize=True))
        st.write(data_analysis.groupby([target]).describe().transpose())
        # Plotting the distribution of the target variable
        plt.figure(figsize=(8, 5))
        sns.countplot(x=target, data=data_analysis, palette=custom_palette[:2])
        plt.title(f'Distribution of {target}')
        plt.xlabel(target)
        plt.ylabel('Count')
        st.pyplot(plt)

#genral analysis with visualizations
st.subheader("General Analysis with Visualizations")

st.write("This section provides general analysis of the dataset with visualizations.")

st.markdown("**Distribution of CKD Status**")
plt.figure(figsize=(8,5))
sns.countplot(data = data_analysis,x="Dialysis_Needed",hue = "CKD_Status",palette=custom_palette)
plt.title("Dialysis Needed by CKD Status")
plt.xlabel("Dialysis Needed")
plt.ylabel("Count")
plt.legend(title="CKD Status")
st.pyplot(plt)


bins =[0,18,35,50,65,100]
labels =['0-18','19-35','36-50','51-65','66+']

st.markdown("**Distribution of Age Groups**")
data_analysis['Age_Group'] = pd.cut(data_analysis['Age'], bins=bins, labels=labels, right=False)
plt.figure(figsize=(12, 6))
sns.countplot(data=data_analysis, x='Age_Group', palette=blue_palette)
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
st.pyplot(plt)

#correlation for numeric features
numeric_data = data_analysis.select_dtypes(include=[np.number])
st.markdown("**Correlation Analysis**")
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
st.pyplot(plt)

page = st.sidebar.selectbox("Choose Page", [
    "CKD Status Analysis",
    "Dialysis Needed Analysis",
    "Feature Engineering & Selection",
    "Model Training & Evaluation",
    "Model Deployment"
])
st.title(f"Visualization for {page}")
if page =="CKD Status Analysis":
    st.write("This section provides analysis of CKD status in the dataset.")
    col1,col2 = st.columns(2)
    
    with col1:
        st.markdown("**CKD Status by Age Group**")
        fig1, ax1 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Age_Group', hue='CKD_Status', palette=custom_palette[:2],ax=ax1)
        plt.title('CKD Status by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    with col2:
        st.markdown("**CKD Status by Diabetes Status**")
        fig2,ax2= plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Diabetes', hue='CKD_Status', palette=custom_palette[2:4],ax=ax2)
        plt.title('Diabetes Status by CKD Status')
        plt.xlabel('Diabetes Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='CKD Status')
        st.pyplot(fig2)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**CKD Status by Hypertension Status**")
        fig3, ax3 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Hypertension', hue='CKD_Status', palette=custom_palette[4:6], ax=ax3)
        plt.title('Hypertension Status by CKD Status')
        plt.xlabel('Hypertension Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='CKD Status')
        st.pyplot(fig3)

    with col4:
        st.markdown("**CKD Status by BUN Levels**")
        fig4, ax4 = plt.subplots(figsize=(14, 10))
        sns.boxplot(data=data_analysis,x='CKD_Status',y='BUN',palette=custom_palette[:2])
        plt.title('BUN Levels by CKD Status')
        plt.xlabel('CKD Status')
        plt.ylabel('BUN Levels')
        plt.xticks(rotation=45)
        st.pyplot(fig4)

    col5,col6 = st.columns(2)
    with col5:
        st.markdown("**CKD Status by Creatinine Levels**")
        fig5, ax5 = plt.subplots(figsize=(14, 10))
        sns.boxplot(data=data_analysis,x='CKD_Status',y='Creatinine_Level',palette=custom_palette[2:4], ax=ax5)
        plt.title('Creatinine Levels by CKD Status')
        plt.xlabel('CKD Status')
        plt.ylabel('Creatinine Levels')
        plt.xticks(rotation=45)
        st.pyplot(fig5)
    with col6:
        st.markdown("**CKD Status by Urine Output Levels**")
        fig6, ax6 = plt.subplots(figsize=(14, 10))
        sns.boxplot(data=data_analysis,x='CKD_Status',y='Urine_Output',palette=custom_palette[6:8], ax=ax6)
        plt.title('Urine Output Levels by CKD Status')
        plt.xlabel('CKD Status')
        plt.ylabel('Urine Output Levels')
        plt.xticks(rotation=45)
        st.pyplot(fig6)
elif page == "Dialysis Needed Analysis":
    st.write("This section provides analysis of dialysis needed in the dataset.")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dialysis Needed by Age Group**")
        fig7, ax7 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Age_Group', hue='Dialysis_Needed', palette=custom_palette[6:8], ax=ax7)
        plt.title('Dialysis Needed by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig7)

    with col2:
        st.markdown("**Dialysis Needed by Diabetes Status**")
        fig8, ax8 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Diabetes', hue='Dialysis_Needed', palette=custom_palette[2:4], ax=ax8)
        plt.title('Diabetes Status by Dialysis Needed')
        plt.xlabel('Diabetes Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Dialysis Needed')
        st.pyplot(fig8)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Dialysis Needed by Hypertension Status**")
        fig9, ax9 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Hypertension', hue='Dialysis_Needed', palette=custom_palette[6:8], ax=ax9)
        plt.title('Hypertension Status by Dialysis Needed')
        plt.xlabel('Hypertension Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Dialysis Needed')
        st.pyplot(fig9)

    with col4:
        st.markdown("**Dialysis Needed by BUN Levels**")
        fig10, ax10 = plt.subplots(figsize=(14, 10))
        sns.boxplot(data=data_analysis,x='Dialysis_Needed',y='BUN',palette=custom_palette[:2], ax=ax10)
        plt.title('BUN Levels by Dialysis Needed')
        plt.xlabel('Dialysis Needed')
        plt.ylabel('BUN Levels')
        plt.xticks(rotation=45)
        st.pyplot(fig10)

elif page == "Feature Engineering & Selection":
    st.title("Feature Engineering & Selection")
    st.write("This section provides insights into feature engineering and selection.")

    st.markdown("**1. Current Feature List**")
    st.write(data_analysis.columns.tolist())

    st.markdown("**2. Missing Values**")
    st.write(data_analysis.isnull().sum())

    st.markdown("**3. Feature Selection**")
    selected_features = ['Age', 'BUN', 'Creatinine_Level', 'Urine_Output', 'Diabetes', 'Hypertension', 'GFR']
    st.write("Selected for modeling:", selected_features)

    st.markdown("**4. Feature Engineering**")
    data_eng = data_analysis.copy()
    data_eng['BUN_Category'] = pd.cut(data_eng['BUN'], bins=[0, 20, 40, 60, 80, 100], 
                                      labels=['Low', 'Medium', 'High', 'Very High', 'Extreme'], right=False)
    data_eng['Creatinine_Category'] = pd.cut(data_eng['Creatinine_Level'], bins=[0, 1, 2, 3, 4, 5], 
                                             labels=['Normal', 'Mild', 'Moderate', 'Severe', 'Critical'], right=False)
    data_eng['GFR_category'] = pd.cut(data_eng['GFR'], bins=[0, 15, 30, 45, 60, 75, 90, 120],
                                      labels=['Very Low', 'Low', 'Moderate', 'Mild', 'Normal', 'High', 'Very High'], right=False)
    data_eng['Age_Group'] = pd.cut(data_eng['Age'], bins=[0, 20, 40, 60, 80, 100], 
                                   labels=['0–20', '21–40', '41–60', '61–80', '81+'], right=False)

    engineered_features = ['Age_Group', 'BUN_Category', 'Creatinine_Category', 'GFR_category']
    st.write("Engineered:", engineered_features)

    dropped_features = ['BUN', 'Creatinine_Level', 'Urine_Output', 'Dialysis_Needed', 'Age', 'GFR']
    data_eng.drop(columns=dropped_features, inplace=True)

    st.markdown("**5. Final Feature List**")
    st.write(data_eng.columns.tolist())

    st.markdown("**Preview of Engineered Data**")
    st.dataframe(data_eng.head())

    # Store for next page
    st.session_state['data_for_model'] = data_eng.copy()
elif page == "Model Training & Evaluation":
    if 'data_for_model' in st.session_state:
        model_data = st.session_state['data_for_model'].copy()
    else:
        st.info("Feature engineered data not found. Generating on the fly...")
        model_data = data_analysis.copy()

        # Reapply feature engineering
        model_data['BUN_Category'] = pd.cut(model_data['BUN'], bins=[0, 20, 40, 60, 80, 100], 
                                            labels=['Low', 'Medium', 'High', 'Very High', 'Extreme'], right=False)
        model_data['Creatinine_Category'] = pd.cut(model_data['Creatinine_Level'], bins=[0, 1, 2, 3, 4, 5], 
                                                labels=['Normal', 'Mild', 'Moderate', 'Severe', 'Critical'], right=False)
        model_data['GFR_category'] = pd.cut(model_data['GFR'], bins=[0, 15, 30, 45, 60, 75, 90, 120],
                                            labels=['Very Low', 'Low', 'Moderate', 'Mild', 'Normal', 'High', 'Very High'], right=False)
        model_data['Age_Group'] = pd.cut(model_data['Age'], bins=[0, 20, 40, 60, 80, 100], 
                                        labels=['0–20', '21–40', '41–60', '61–80', '81+'], right=False)

        # Drop unused columns
        model_data.drop(columns=['BUN', 'Creatinine_Level', 'Urine_Output', 'Dialysis_Needed', 'Age', 'GFR'], inplace=True)

    st.markdown("**1. Label Encoding**")
    le = LabelEncoder()
    for col in ['Diabetes', 'Hypertension', 'CKD_Status', 'BUN_Category', 'Creatinine_Category', 'Age_Group', 'GFR_category']:
        if col in model_data.columns:
            model_data[col] = le.fit_transform(model_data[col])

    st.write("Encoded data preview:")
    st.dataframe(model_data.head())

    st.markdown("**2. Data Splitting**")
    X = model_data.drop(columns=['CKD_Status'])
    y = model_data['CKD_Status']
    st.write("Features (X):", X.columns.tolist())
    st.write("Target (y): CKD_Status")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.markdown("**4. Model Training and Evaluation**")
    st.write("Select models to train and evaluate:")

    models = st.sidebar.multiselect("Select Models to Train",
        ["Logistic Regression", "Decision Tree", "Random Forest"], default=["Logistic Regression"])
    
    if models:
        # Dictionary to store results for plotting
        model_results = {}
        classification_reports = {}
        
        # Train all selected models and store results
        for model in models:
            if model == "Logistic Regression":
                lr_model = LogisticRegression()
                lr_model.fit(X_train, y_train)
                lr_preds = lr_model.predict(X_test)
                model_results[model] = {
                    'predictions': lr_preds,
                    'accuracy': accuracy_score(y_test, lr_preds),
                    'confusion_matrix': confusion_matrix(y_test, lr_preds)
                }
                classification_reports[model] = classification_report(y_test, lr_preds, output_dict=True)
                
            elif model == "Decision Tree":
                dt_model = DecisionTreeClassifier()
                dt_model.fit(X_train, y_train)
                dt_preds = dt_model.predict(X_test)
                model_results[model] = {
                    'predictions': dt_preds,
                    'accuracy': accuracy_score(y_test, dt_preds),
                    'confusion_matrix': confusion_matrix(y_test, dt_preds)
                }
                classification_reports[model] = classification_report(y_test, dt_preds, output_dict=True)
                
            elif model == "Random Forest":
                rf_model = RandomForestClassifier()
                rf_model.fit(X_train, y_train)
                rf_preds = rf_model.predict(X_test)
                model_results[model] = {
                    'predictions': rf_preds,
                    'accuracy': accuracy_score(y_test, rf_preds),
                    'confusion_matrix': confusion_matrix(y_test, rf_preds)
                }
                classification_reports[model] = classification_report(y_test, rf_preds, output_dict=True)

        # Display accuracy scores
        st.subheader("Model Accuracy Comparison")
        accuracy_df = pd.DataFrame([
            {"Model": model, "Accuracy": f"{results['accuracy']:.4f}"} 
            for model, results in model_results.items()
        ])
        st.dataframe(accuracy_df, use_container_width=True)

        # Display confusion matrices in columns
        st.subheader("Confusion Matrices")
        num_models = len(models)
        if num_models == 1:
            cm = model_results[models[0]]['confusion_matrix']
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
            plt.title(f'Confusion Matrix - {models[0]}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(plt)
        else:
            cols = st.columns(num_models)
            for i, (model_name, results) in enumerate(model_results.items()):
                with cols[i]:
                    cm = results['confusion_matrix']
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
                    plt.title(f'{model_name}')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(plt)

        # Display classification reports as markdown tables
        st.subheader("Classification Reports")
        for model_name, report in classification_reports.items():
            st.markdown(f"**{model_name} Classification Report**")
            
            # Convert classification report to markdown table
            report_data = []
            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    class_label = 'Not CKD' if class_name == '0' else 'CKD' if class_name == '1' else class_name
                    report_data.append({
                        'Class': class_label,
                        'Precision': f"{metrics['precision']:.3f}",
                        'Recall': f"{metrics['recall']:.3f}",
                        'F1-Score': f"{metrics['f1-score']:.3f}",
                        'Support': int(metrics['support'])
                    })
            
            # Add summary rows
            if 'macro avg' in report:
                report_data.append({
                    'Class': 'Macro Avg',
                    'Precision': f"{report['macro avg']['precision']:.3f}",
                    'Recall': f"{report['macro avg']['recall']:.3f}",
                    'F1-Score': f"{report['macro avg']['f1-score']:.3f}",
                    'Support': int(report['macro avg']['support'])
                })
            
            if 'weighted avg' in report:
                report_data.append({
                    'Class': 'Weighted Avg',
                    'Precision': f"{report['weighted avg']['precision']:.3f}",
                    'Recall': f"{report['weighted avg']['recall']:.3f}",
                    'F1-Score': f"{report['weighted avg']['f1-score']:.3f}",
                    'Support': int(report['weighted avg']['support'])
                })
            
            # Display as markdown table
            report_df = pd.DataFrame(report_data)
            st.markdown(report_df.to_markdown(index=False))
            st.markdown("---")  # Separator between models

            st.markdown("**Model Saving**")
            save_model = st.checkbox("Save Best Model")
            if save_model:
                best_model_name = max(model_results, key=lambda x: model_results[x]['accuracy'])
                best_model = None
                if best_model_name == "Logistic Regression":
                    best_model = lr_model
                elif best_model_name == "Decision Tree":
                    best_model = dt_model
                elif best_model_name == "Random Forest":
                    best_model = rf_model
                
                if best_model:
                    with open(f"{best_model_name.replace(' ', '_').lower()}_model.pkl", 'wb') as f:
                        pickle.dump(best_model, f)
                    st.success(f"{best_model_name} saved successfully!")
