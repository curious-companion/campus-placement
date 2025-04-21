import gradio as gr
import pickle
import numpy as np

#Load the models
with open('models/Placed_model.pkl', 'rb') as f:
    placed_model = pickle.load(f)

with open('models/sal_model.pkl', 'rb') as f:
    sal_model = pickle.load(f)

def predict_placement(features):
    try:
        features = np.array(features).reshape(1,-1)
        prediction = placed_model.predict(features)
        return prediction[0]
    except:
        return "Invalid Input"

def predict_salary(features):
    try:
        features = np.array(features).reshape(1,-1)
        prediction = sal_model.predict(features)
        return prediction[0]
    except:
        return "Invalid Input"

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Tab("Placement Prediction"):
        gr.Markdown("## Predict Placement Status")
        placement_input = gr.Textbox(label="Placement Prediction")
        placement_output = gr.Textbox(label="Placement Prediction Result")
        gr.button("Predict Placement").click(predict_placement, inputs=placement_input, outputs=placement_output)
    
    with gr.Tab("Salary Prediction"):
        gr.Markdown("## Predict Salary")
        salary_input = gr.Textbox(label="Salary Prediction")
        salary_output = gr.Textbox(label="Salary Prediction Result")
        gr.button("Predict Salary").click(predict_salary, inputs=salary_input, outputs=salary_output)

demo.launch()