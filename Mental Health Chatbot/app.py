from flask import Flask, request, jsonify, render_template
import os
from openai import OpenAI
from dotenv import load_dotenv

#Load environment variables
load_dotenv()

#Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

#OpenAI client setup
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#System prompt for 4o-mini
system_prompt = """
You are a mental health assistant chatbot.

Rules:
- Your job is to provide mental health support, encouragement, and coping strategies only.
- If the user's message is unrelated to mental health (e.g. asking for jokes, general knowledge, math problems, technology help, etc.), politely remind them: 
  'I'm here to help with mental health support. Please send a message related to how you are feeling or what is on your mind.'
- If the user sends 3 consecutive irrelevant messages, reset the conversation by saying:
  'I'm only able to continue if we focus on your mental health. Let's start fresh. How are you feeling today?'

Behavior:
- For relevant messages: 
    - Read carefully
    - Guess the user's issue (anxiety, depression, stress, PTSD, etc.)
    - Respond supportively in a paragraph
    - Offer 3-4 helpful coping strategies

- Always be supportive, positive, and professional.
- Never diagnose conditions.
- Encourage seeking professional help if needed and provide them with links or contact numbers for UK support services.
"""


#Welcome message
welcome_message = "Hello! How are you feeling today? What would you like to talk about today?"

#Home page
@app.route('/')
def home():
    return render_template('index.html')

#Chat route
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message received.'}), 400

    try:
        #if frontend sends "__WELCOME__", reply with static welcome
        if user_message == "__WELCOME__":
            return jsonify({'response': welcome_message})

        #Otherwise normal OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.4,
            max_tokens=800
        )

        assistant_message = response.choices[0].message.content.strip()

        return jsonify({'response': assistant_message})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

#Run app
if __name__ == '__main__':
    app.run(debug=True)
