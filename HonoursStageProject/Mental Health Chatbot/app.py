from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session
client = OpenAI(
  api_key="OpenAIKey" #Hidden for now whilst uploading to github
)

@app.route('/')
def home():
    if 'messages' not in session:
        session['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    if 'messages' not in session:
        session['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]
    
    # Add user message to conversation history
    session['messages'].append({"role": "user", "content": user_message})

    try:
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=session['messages'],
            temperature=0.7,
            max_tokens=2048
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to conversation history
        session['messages'].append({"role": "assistant", "content": assistant_message})
        
        return jsonify({'response': assistant_message})
    
    except Exception as e:
        error_message = str(e)
        if 'insufficient_quota' in error_message:
            return jsonify({'error': 'OpenAI API quota exceeded. Please check your API key or billing details.'}), 500
        elif 'invalid_api_key' in error_message:
            return jsonify({'error': 'Invalid API key. Please check your API key configuration.'}), 500
        else:
            return jsonify({'error': f'An error occurred: {error_message}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
