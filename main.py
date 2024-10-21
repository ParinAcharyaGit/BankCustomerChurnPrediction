import pandas as pd
import streamlit as st
import pickle
import numpy as np
import os
from openai import OpenAI
from utils import create_gauge_chart, create_model_probability_chart
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt

client = OpenAI(
  base_url = 'https://api.groq.com/openai/v1',
  api_key = os.environ['GROQ_API_KEY']
)

def explain_prediction(probability, input_dict, surname):
  prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer with the surname {surname} has a {round(probability*100, 1)}% probability of churning, based on the information
  provided below.

  Here is the customer's information: {input_dict}

  Here are the machine learning model's most important features for predicting churn in descending order:

            Feature    |    Importance
    ---------------------------------------
    NumOfProducts      |    0.323888
    IsActiveMember     |    0.164146
    Age                |    0.109550
    Geography_Germany  |    0.091373
    Balance            |    0.052786
    Geography_France   |    0.046463
    Gender_Female      |    0.045283
    Geography_Spain    |    0.036855
    CreditScore        |    0.035005
    EstimatedSalary    |    0.032655
    HasCrCard          |    0.031940
    Tenure             |    0.030054
    Gender_Male        |    0.000000

    {pd.set_option('display.max_columns', None)}

    Here are summary statistics for churned customers:
    {df[df['Exited'] == 1].describe()}

    Here are summary statistics for non-churned customers:
    {df[df['Exited'] == 0].describe()}

    
    Base your response on the following criteria depending on the churn probability:
    - If the customer has over 40% risk of churning, generate a brief maximum 3 sentence explanation of why they are at risk of churning.
    - If the customer has less than 40% risk of churning, generate a brief maximum 3 sentence explanation of why they are not at risk of churning.
    - Your explanation should be base on the customer's information, the summary statistics of churned and non-churned customers, and the machine learning model's most important features as provided.
    - Your explanation should not include any other properties of the customers, other than the ones mentioned.

    Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's predictions and the most important features ...", just explain the prediction.
    
        
  """

  print("EXPLANATION_PROMPT", prompt)
  
  raw_response = client.chat.completions.create(
    model = "llama-3.2-3b-preview",
    messages = [{
      "role": "user",
      "content" : prompt
    }]
  )
  
  return raw_response.choices[0].message.content

def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)

# Initialising the models

xgboost_model = load_model('xgb_model.pkl')

naive_bayes_model = load_model('nb_model.pkl')

random_forest_model = load_model('rf_model.pkl')

decision_tree_model = load_model('dt_model.pkl')

svm_model = load_model('svm_model.pkl')

knn_model = load_model('knn_model.pkl')

voting_classifier_model = load_model('voting_classifier.pkl')

xgboost_SMOTE_model = load_model('xgboost_smote.pkl')

xgboost_feature_engineered_model = load_model('xgboost_feature_engineered.pkl')

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, 
                  is_active_member, estimated_salary):
  input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumberofProducts': num_products,
    "HasCreditCard": int(is_active_member),
    'IsActiveMember': int(is_active_member),
    'EstimatedSalary': estimated_salary,
    "Geography_France": 1 if location == "France" else 0,
    "Geography_Germany": 1 if location == "Germany" else 0,
    "Geography_Spain": 1 if location == "Spain" else 0,
    "GenderMale": 1 if gender == "Male" else 0,
    "GendereFemale": 1 if gender == "Female" else 0
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_predictions(input_df, input_dict):

  probabilities = {
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
    'RandomForest': random_forest_model.predict_proba(input_df)[0][1],
    'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
  }

  avg_probability = np.mean(list(probabilities.values()))

  with col1:
    fig = create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

  with col2:
    fig_probs = create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  return avg_probability     

# Generate a personalized email
def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""You are a manager at a HS Bank. You are responsible for ensuring that customers stay with the bank and are incentivised with various offers.

  You noticed a customer names {surname} has a {round(probability * 100, 1)}% probability of churning.

  Here is the customer's information: {input_dict}

  Here is some explanation as to why the customer migh be at risk of churning: {explanation}

  Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them with incentives so that they become more loyal woth the bank.

  Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model, to the customer.
  """
  raw_response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
      {
        "role": "user",
        "content": prompt
      },
    ],    
  )
  print("\n\nEMAIL PROMPT", prompt)

  return raw_response.choices[0].message.content
  
# Display information
st.title("Bank Customer Churn Prediction📊")
st.markdown("Created by [Parin Acharya](https://www.linkedin.com/in/parinacharya/)")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print(selected_customer_id)
  selected_surname = selected_customer_option.split(" - ")[1]
  print(selected_surname)

  selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
  print("Selected customer", selected_customer)
  
  col1, col2 = st.columns(2)

  with col1:    
    credit_score = st.number_input(
      "Credit Score",
      min_value = 300,
      max_value = 800,
      value = int(selected_customer['CreditScore'])
    )

    countries = ['Spain', 'France', 'Germany']
  
    location = st.selectbox(
      "Location", countries,
      index = countries.index(
        selected_customer['Geography']
      )
    )
  
    gender = st.radio("Gender", ["Male", "Female"], index = 0 if selected_customer['Gender'] == "Male" else 1)
  
    age = st.number_input(
      "Age",
      min_value = 18,
      max_value= 100,
      value = int(selected_customer['Age'])
    )

    tenure = st.number_input(
      "Tenure {years}",
      min_value = 0,
      max_value = 50,
      value = int(selected_customer['Tenure'])
    )

  with col2:    
    balance = st.number_input(
      "Balance",
      min_value = 0,
      value = int(selected_customer['Balance'])
    )

    num_products = st.number_input(
      "Number of Products",
      min_value = 1,
      max_value = 10,
      value = int(selected_customer['NumOfProducts'])
    )

    has_credit_card = st.checkbox("Has Credit Card", value = bool(selected_customer['HasCrCard']))

    is_active_member = st.checkbox("Is Active Member", value = bool(selected_customer['IsActiveMember']))

    estimated_salary = st.number_input(
      "Estimated Salary",
      min_value = 0.0,
      value = float(selected_customer['EstimatedSalary'])
    )

  predict = st.button("Will this customer churn?")
  if predict:

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, 
                                          is_active_member, estimated_salary)
      
    avg_probability = make_predictions(input_df, input_dict)
    st.markdown("---")
    st.subheader("Churn Probability Result")
    st.write(f"Average Probability of Churn for {selected_customer['Surname']} : {round(avg_probability*100, 1)}%")
    
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
    st.markdown("---")
    st.subheader("Explanation of the Prediction")
    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
    st.markdown("---")
    st.subheader("Email to Customer")
    st.markdown(email)

    st.markdown("---")  
    st.markdown("Customer percentiles for different metrics")
    metric_options = ['Balance', 'CreditScore', 'Tenure', 'EstimatedSalary']
    selected_metric = st.selectbox('Select a metric to display percentiles:', metric_options)

    # Locate the user by their surname and find their balance directly
    user_value = df.loc[df['Surname'] == selected_surname, selected_metric].values[0]

    # Calculate the percentile of the user's value compared to the entire selected metric column
    user_percentile = percentileofscore(df[selected_metric], user_value)

    # Display the user's value and percentile rank
    st.write(f"The {selected_metric.lower()} for {selected_surname} is: {user_value}")
    st.write(f"The customer falls in the {user_percentile:.2f}th percentile of {selected_metric.lower()}.")

    # Create the pie chart
    percentile_data = [user_percentile, 100 - user_percentile]  # User's percentile and the rest
    labels = [f"{user_percentile:.2f}% of customers", f"{100 - user_percentile:.2f}% of customers"]

    fig, ax = plt.subplots()
    ax.pie(percentile_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'])
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is drawn as a circle.

    # Display the pie chart in Streamlit
    st.pyplot(fig)


    
  st.markdown("---")  
  st.markdown("Thank you for using this tool")

      



    
      

      

                                                                      
                                                          
                                                                      

    
    


  


