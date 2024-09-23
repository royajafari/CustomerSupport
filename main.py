from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the model
model = joblib.load("text_classifier.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI & Gradio Integration!"}

@app.get("/predict/")
def predict(input_text: str):
    # Use the model to predict the label
    output = model.predict([input_text])  # Make a prediction
    return {"output": output[0]}  # Return the predicted label

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
