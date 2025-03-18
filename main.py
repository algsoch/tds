import os
import pandas as pd
import json
import base64
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from collections import Counter
from jinja2 import Template
from difflib import get_close_matches
from dotenv import load_dotenv
import google.generativeai as genai  # Using Gemini API

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Initialize FastAPI app
app = FastAPI()

# Load dataset
dataset_path = "training_dataset.csv"
df = pd.read_csv(dataset_path)
df["question"] = df["question"].str.strip().str.lower()  # Normalize dataset questions

# Question log to track most asked questions
question_log = []

class QuestionRequest(BaseModel):
    question: str

# Function to find the best matching question in the dataset
def get_best_match(question):
    matches = get_close_matches(question, df["question"].tolist(), n=1, cutoff=0.7)
    if matches:
        return df[df["question"] == matches[0]].iloc[0]["answer"]
    return None

# API endpoint for answering questions
@app.post("/api/question")
def ask_question(request: Request, question: str = Form(...)):
    user_ip = request.client.host  # Get user IP
    question_log.append(question)
    question_normalized = question.strip().lower()
    
    # Try finding an answer in the dataset
    answer = get_best_match(question_normalized)
    
    if not answer:
        answer = "No answer found in the dataset."
    
    image_data = None
    
    # Check if answer is an image file
    if answer.lower().endswith((".png", ".jpg", ".jpeg", ".gif")) and os.path.exists(answer):
        with open(answer, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    response_data = {
        "question": question,
        "matched_question": question_normalized if answer else "No match found",
        "answer": answer if not image_data else "(Image attached)",
        "image": image_data if image_data else None
    }
    
    return JSONResponse(content=response_data)

# Enhanced GUI Interface
template = Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Question Answering</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; background-color: #121212; color: #ffffff; }
        h2 { color: #f0f0f0; }
        form { background: #1e1e1e; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.2); max-width: 500px; margin: auto; }
        input, button { padding: 10px; margin: 10px; width: 90%; border-radius: 5px; border: 1px solid #ddd; }
        input { background-color: #333; color: white; }
        button { background-color: #007bff; color: white; cursor: pointer; transition: 0.3s; }
        button:hover { background-color: #0056b3; }
        .answer-box { background: #1e1e1e; padding: 15px; border-radius: 10px; margin-top: 20px; display: inline-block; text-align: left; }
        .answer-title { font-weight: bold; color: #76c7c0; font-size: 18px; }
        .copy-btn { cursor: pointer; padding: 10px; background-color: #28a745; color: white; border: none; border-radius: 5px; transition: 0.3s; }
        .copy-btn:hover { background-color: #218838; }
        ul { list-style: none; padding: 0; }
        li { background: #1e1e1e; padding: 10px; margin: 5px; border-radius: 5px; box-shadow: 0px 0px 5px rgba(255, 255, 255, 0.1); }
        img { max-width: 100%; border-radius: 5px; margin-top: 10px; }
    </style>
    <script>
        function askQuestion(event) {
            event.preventDefault();
            let question = document.getElementById("questionInput").value;
            document.getElementById("loading").style.display = "block";
            fetch("/api/question", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: "question=" + encodeURIComponent(question)
            })
            .then(response => response.json())
            .then(data => {
                let imageHtml = data.image ? `<img src='data:image/png;base64,${data.image}' alt='Answer Image'>` : "";
                document.getElementById("answerBox").innerHTML = `
                    <p class='answer-title'>Matched Question:</p>
                    <p>${data.matched_question}</p>
                    <p class='answer-title'>Answer:</p>
                    <p>${data.answer}</p>
                    ${imageHtml}
                `;
                document.getElementById("answerSection").style.display = "block";
            })
            .catch(error => alert("Error fetching answer."))
            .finally(() => {
                document.getElementById("loading").style.display = "none";
            });
        }
    </script>
</head>
<body>
    <h2>Ask a Question</h2>
    <form id="questionForm" onsubmit="askQuestion(event)">
        <input type="text" id="questionInput" name="question" placeholder="Enter your question...">
        <button type="submit">Ask</button>
    </form>
    <div id="loading" style="display: none;">Fetching answer...</div>
    <h3>Most Asked Questions:</h3>
    <ul>
        {% for q in most_asked %}
            <li>{{ q }}</li>
        {% endfor %}
    </ul>
    <div id="answerSection" style="display: none; margin-top: 20px;">
        <div class="answer-box" id="answerBox"></div>
    </div>
</body>
</html>
''')

@app.get("/", response_class=HTMLResponse)
def home():
    most_asked = [q[0] for q in Counter(question_log).most_common(5)]
    return template.render(most_asked=most_asked)
