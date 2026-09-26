from flask import Flask, render_template, request
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Base directory of the current file
BASE_DIR = os.path.dirname(__file__)
intents_path = os.path.join(BASE_DIR, 'dataset', 'intents1.json')

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set. Please set it in a .env file.")

# Load intents and prepare the system instruction
with open(intents_path, 'r') as f:
    intents = json.load(f)

system_instruction = (
    "You are an intelligent and helpful chatbot for the Royal College of Engineering. "
    "Use the following knowledge base to answer student and parent queries accurately. "
    "If the answer is not in the knowledge base, be polite and say you don't have that information. "
    "Keep your answers concise and friendly.\n\nKnowledge Base:\n"
)

for intent in intents['intents']:
    tag = intent.get('tag', '')
    responses = " ".join(intent.get('responses', []))
    system_instruction += f"- Topic ({tag}): {responses}\n"

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def chatbot_response(user_input):
    if not API_KEY:
        return "System error: Gemini API key is missing. Please create a `.env` file in the root directory and add `GEMINI_API_KEY=your_key_here`."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_input}]}
        ]
    }
    
    try:
        response = requests.post(url, json=payload, verify=False)
        response.raise_for_status()
        data = response.json()
        
        # Extract the text from the response
        try:
            return data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return "Sorry, I received an unexpected response format from the AI."
            
    except requests.exceptions.RequestException as e:
        return f"An error occurred while communicating with the AI: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
