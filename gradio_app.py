import gradio as gr
import requests

# Gradio function that sends a request to FastAPI for predictions
def gradio_fn(input_text):
    try:
        # Sending a request to FastAPI (assuming FastAPI is running on port 8000)
        response = requests.get(f"http://127.0.0.1:8001/predict/", params={"input_text": input_text})
        if response.status_code == 200:
            return response.json()["output"]
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return str(e)

# Gradio interface
def create_gradio_interface():
    interface = gr.Interface(
        fn=gradio_fn,
        inputs=gr.Textbox(label="Input Text"),
        outputs="text",
        title="Text Classification Model",
    )
    interface.launch(share=True)

if __name__ == "__main__":
    create_gradio_interface()
