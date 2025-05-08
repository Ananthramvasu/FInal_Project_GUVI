import pickle
import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System")
task = st.sidebar.radio("Select Prediction Task", ["Fraudulent_Claim_Predictions", "Risk_Score_Predictions","Sentiment_Prediction","Claim_Amount_Prediction"])
base_path = "C:/Users/Ananthram Vasu/Downloads/insurance_dataset/"
with open(base_path + 'Fraudulent_Claim_Predictions.pkl', 'rb') as file:
    fcp = pickle.load(file)

with open(base_path + 'Risk_Score_Predictions.pkl', 'rb') as file:
    rsp = pickle.load(file)

#with open("C:/Users/Ananthram Vasu/Downloads/Final - 2 - Insurance Review Project/Final_Project_sentiment.pkl", 'rb') as file:
    #fps = pickle.load(file)

#with open("C:/Users/Ananthram Vasu/Downloads/Final - 2 - Insurance Review Project/count_vectorizer.pkl", 'rb') as file:
    #cv = pickle.load(file)

with open(base_path + 'Claim_Amount_Prediction.pkl', 'rb') as file:
    cmp = pickle.load(file)

def common_inputs():
    Vehicle_Age = st.number_input("Enter your Vehicle Age:")
    Customer_Age = st.number_input("Enter your age: ")
    Airbags = st.slider("Select the airbags", 0,10,2)
    Displacement = st.number_input("Enter your Car's Displacement in normal terms(eg:1197)")
    Cylinder = st.number_input("Enter the number of cylinders based on your displacement")
    Gross_Weight = st.number_input("Enter the gross weight of your car")
    Annual_Income = st.number_input("Enter your Annual Income")
    Insurance_Premium = st.number_input("Enter your annual insurance premium amount")
    Claim_History = st.number_input("Enter your Claim History in numbers")
    Segment = st.selectbox("Enter your Segment: If A = 0, B1 = 1,B2 = 3,C1 = 4,C2 = 5,Utility = 6",[0,1,2,3,4,5,6])
    Model = st.selectbox("Select your Model if M1 = 1,M2 = 2,M3 = 3,M4 = 4,M5 = 5,M6 = 6,M7 = 7,M8 = 8,M9 = 9,M10 = 10,M11 = 11",[1,2,3,4,5,6,7,8,9,10,11])
    Fuel_Type = st.selectbox("Select the Fuel Type of your car if Petrol = 0, Disel = 1, Gas(LPG,CNG) = 2",[0,1,2])
    Gender = st.selectbox("Gender: Male = 0, Female = 1", [0,1])
    Policy_Type = st.selectbox("Select your Policy Type if Health = 0, Life Insurance = 1, Automobile = 2,Property = 3",[0,1,2,3])
    Max_Torque_Nm = st.number_input("Enter the maximum torque yor car produce in nm")
    Max_Power_bhp = st.number_input("Enter the maximum power your car produce in bhp")
    return Vehicle_Age, Customer_Age, Airbags, Displacement, Cylinder, Gross_Weight, Annual_Income, Insurance_Premium, Claim_History, Segment, Model, Fuel_Type, Gender, Policy_Type, Max_Torque_Nm, Max_Power_bhp

    
    
if task == 'Fraudulent_Claim_Predictions':
    st.subheader("Fraudulent Claim Prediction")
    Vehicle_Age, Customer_Age,Airbags, Displacement, Cylinder,Gross_Weight, Annual_Income, Insurance_Premium, Claim_History,Segment, Model, Fuel_Type,Gender, Policy_Type, Max_Torque_Nm,Max_Power_bhp = common_inputs()
    Claim_History_Label = st.selectbox("Select the label based on your Claim History if No Prior Claims = 0,Low History = 1,High History = 2",[0,1,2])
    
    #build the inputs as a dataframe
    inp_dict = {
    'Vehicle_Age': Vehicle_Age,
    'Customer_Age': Customer_Age,
    'Airbags': Airbags,
    'Displacement': Displacement,
    'Cylinder': Cylinder,
    'Gross_Weight': Gross_Weight,
    'Annual_Income': Annual_Income,
    'Insurance_Premium': Insurance_Premium,
    'Claim_History': Claim_History,
    'Segment': Segment,
    'Model': Model,
    'Fuel_Type': Fuel_Type,
    'Claim_History_Label': Claim_History_Label,
    'Gender': Gender,
    'Policy_Type': Policy_Type,
    'Max_Torque_Nm': Max_Torque_Nm,
    'Max_Power_bhp': Max_Power_bhp 
    
    }
    
    if st.button("Predict"):
        input_df = pd.DataFrame([inp_dict])
        prediction = fcp.predict(input_df)

        
        if prediction[0] == 1:
            st.success("Fraudulent Claim (Fraud detected)")
        else:
            st.success("Genuine Claim (Not Fraud)")

    
    
elif task == 'Risk_Score_Predictions':
    st.subheader("Risk_Score_Predictions")
    Vehicle_Age, Customer_Age,Airbags, Displacement, Cylinder,Gross_Weight, Annual_Income, Insurance_Premium, Claim_History,Segment, Model, Fuel_Type,Gender, Policy_Type, Max_Torque_Nm,Max_Power_bhp = common_inputs()
    
    in_di = {
    'Vehicle_Age': Vehicle_Age,
    'Customer_Age': Customer_Age,
    'Airbags': Airbags,
    'Displacement': Displacement,
    'Cylinder': Cylinder,
    'Gross_Weight': Gross_Weight,
    'Annual_Income': Annual_Income,
    'Insurance_Premium': Insurance_Premium,
    'Claim_History': Claim_History,
    'Segment': Segment,
    'Model': Model,
    'Fuel_Type': Fuel_Type,
    'Gender': Gender,
    'Policy_Type': Policy_Type,
    'Max_Torque_Nm': Max_Torque_Nm,
    'Max_Power_bhp': Max_Power_bhp
    }
    
    if st.button("Predict"):
        in_df = pd.DataFrame([in_di])
        prediction = rsp.predict(in_df)

        if prediction == 0:
            st.success("High Risk Predicted (0)")
        elif prediction == 1:
            st.success("Low Risk Predicted (1)")
        elif prediction == 2:
            st.success("Medium Risk Predicted (2)")
    
    
    
elif task == 'Claim_Amount_Prediction':
    st.subheader("Claim_Amount_Prediction")
    Vehicle_Age, Customer_Age,Airbags, Displacement, Cylinder,Gross_Weight, Annual_Income, Insurance_Premium, Claim_History,Segment, Model, Fuel_Type,Gender, Policy_Type, Max_Torque_Nm,Max_Power_bhp = common_inputs()
    Subscription_Length = st.number_input("Enter Your Subscription Length")
    Length = st.number_input("Enter the length of your Vehicle")
    Width = st.number_input("Enter the width of your Vehicle")
    Claim_History_Label = st.selectbox("Select the label based on your Claim History if No Prior Claims = 0,Low History = 1,High History = 2",[0,1,2])
    Risk_Score = st.number_input("First predict your Risk Score. If predicted Enter your Score")
    
    input_di = {
    'Subscription_Length': Subscription_Length,
    'Vehicle_Age': Vehicle_Age,
    'Customer_Age': Customer_Age,
    'Airbags': Airbags,
    'Displacement': Displacement,
    'Cylinder': Cylinder,
    'Length': Length,
    'Width': Width,
    'Gross_Weight': Gross_Weight,
    'Annual_Income': Annual_Income,
    'Insurance_Premium': Insurance_Premium,
    'Claim_History': Claim_History,
    'Claim_History_Label': Claim_History_Label,
    'Risk_Score': Risk_Score,
    'Segment': Segment,
    'Model': Model,
    'Fuel_Type': Fuel_Type,
    'Gender': Gender,
    'Policy_Type': Policy_Type,
    'Max_Torque_Nm': Max_Torque_Nm,
    'Max_Power_bhp': Max_Power_bhp
    
    }
    
    if st.button("Predict"):
        inp_df = pd.DataFrame([input_di])
        prediction = cmp.predict(inp_df)
        st.success(f"Claim Prediction Result: **{prediction}**")
    
elif task == "Sentiment_Prediction":
 # Text input for feedback
    feedback_text = st.text_area("Enter Customer Feedback:")

    # Initialize the sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # Analyze button
    if st.button("Analyze Sentiment"):
        if feedback_text:
            sentiment_score = sentiment_analyzer.polarity_scores(feedback_text)
            compound_score = sentiment_score['compound']
            sentiment_label = (
                "Positive" if compound_score >= 0.05 else
                "Negative" if compound_score <= -0.05 else
                "Neutral"
            )
            st.write(f"Sentiment Score: {compound_score}")
            st.success(f"Predicted Sentiment: {sentiment_label}")
        else:
            st.error("Please enter feedback text to analyze.")
   




   

    
    
    
#'Vehicle_Age', 'Customer_Age','Airbags', 'Displacement', 'Cylinder','Gross_Weight', 'Annual_Income', 'Insurance_Premium', 'Claim_History','Segment', 'Model', 'Fuel_Type','Gender', 'Policy_Type', 'Max_Torque_Nm','Max_Power_bhp'
#'Vehicle_Age', 'Customer_Age','Airbags', 'Displacement', 'Cylinder','Gross_Weight', 'Annual_Income', 'Insurance_Premium', 'Claim_History','Segment', 'Model', 'Fuel_Type','Claim_History_Label','Gender', 'Policy_Type', 'Max_Torque_Nm','Max_Power_bhp'
#'Subscription_Length', 'Vehicle_Age', 'Customer_Age','Airbags', 'Displacement', 'Cylinder', 'Length','Width', 'Gross_Weight', 'Annual_Income', 'Insurance_Premium','Claim_History', 'Claim_History_Label', 'Risk_Score', 'Region_Code','Segment', 'Model', 'Fuel_Type', 'Gender', 'Policy_Type','Max_Torque_Nm', 'Max_Power_bhp'
