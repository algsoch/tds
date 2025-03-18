import os
import pandas as pd
import json
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from collections import Counter
from jinja2 import Template
from difflib import get_close_matches

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

# Function to find the closest matching question
def find_closest_question(question, dataset_questions):
    matches = get_close_matches(question, dataset_questions, n=1, cutoff=0.6)
    return matches[0] if matches else None

# API endpoint for answering questions
@app.post("/api/question")
def ask_question(
    request: Request,
    question: str = Form(...),
    file: UploadFile = File(None)
):
    user_ip = request.client.host  # Get user IP
    question_log.append(question)
    question_normalized = question.strip().lower()
    
    # Find the closest matching question
    closest_question = find_closest_question(question_normalized, df["question"].tolist())
    if closest_question:
        answer_row = df[df["question"] == closest_question]
        answer = answer_row["answer"].values[0]
        if pd.isna(answer) or answer.strip() == "":
            return JSONResponse(content={"answer": "No answer available in the dataset."})
        
        response_data = {
            "question": question,
            "matched_question": closest_question,
            "answer": answer
        }
        
        return JSONResponse(content=response_data)
    
    return JSONResponse(content={"answer": "Question not found in dataset."})

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
    </style>
    <script>
        function askQuestion(event) {
            event.preventDefault();
            let question = document.getElementById("questionInput").value;
            document.getElementById("loading").style.display = "block";
            fetch("/api/question", {
                method: "POST",
                body: new FormData(document.getElementById("questionForm"))
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById("answerBox").innerHTML = `
                    <p class='answer-title'>Matched Question:</p>
                    <p>${data.matched_question}</p>
                    <p class='answer-title'>Answer:</p>
                    <p>${data.answer}</p>
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

# Run the app using uvicorn (Example: uvicorn main:app --reload)
