import pandas as pd
import joblib
import gradio as gr

# Load the trained model and scaler
model = joblib.load('employee_attrition_model.joblib')
scaler = joblib.load('scaler.joblib')

# Prediction function
def predict_attrition(Age, MonthlyIncome, YearsAtCompany, OverTime, JobSatisfaction,
                      DistanceFromHome, NumCompaniesWorked, EnvironmentSatisfaction):
    # Create a dataframe from inputs
    input_df = pd.DataFrame([[Age, MonthlyIncome, YearsAtCompany, OverTime, JobSatisfaction,
                              DistanceFromHome, NumCompaniesWorked, EnvironmentSatisfaction]],
                            columns=['Age','MonthlyIncome','YearsAtCompany','OverTime','JobSatisfaction',
                                     'DistanceFromHome','NumCompaniesWorked','EnvironmentSatisfaction'])
    
    # Convert OverTime to numeric
    input_df['OverTime'] = 1 if OverTime=='Yes' else 0
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Predict
    pred = model.predict(input_scaled)[0]
    
    return "⚠️ Employee is likely to RESIGN" if pred==1 else "✅ Employee is likely to STAY"

# Gradio interface
iface = gr.Interface(
    fn=predict_attrition,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="MonthlyIncome"),
        gr.Number(label="YearsAtCompany"),
        gr.Dropdown(['Yes','No'], label="OverTime"),
        gr.Slider(1,4, step=1, label="JobSatisfaction"),
        gr.Number(label="DistanceFromHome"),
        gr.Number(label="NumCompaniesWorked"),
        gr.Slider(1,4, step=1, label="EnvironmentSatisfaction")
    ],
    outputs="text",
    title="Employee Attrition Prediction"
)

iface.launch()
