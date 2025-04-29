from flask import Flask, request, jsonify, render_template, session
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
You are a compassionate mental health assistant chatbot.

Your role is to support users through difficult emotions such as anxiety, depression, stress, grief, or family issues. You are not a therapist or a doctor — your job is to listen, reflect, and provide gentle, helpful coping suggestions.

Start every conversation with a friendly welcome message like:
"Hello! How are you feeling today? What would you like to talk about?"

When a user expresses emotional distress:
- Gently validate their experience.
- Ask one open-ended question to encourage deeper reflection, e.g. "Would you feel comfortable sharing more about what's been going on?" or "What do you think is contributing to how you're feeling?"

If the user expresses distress clearly and provides at least two emotionally detailed responses in a row, stop asking further probing questions.

Then:
1. Reflect empathetically on what they've shared.
2. Offer 3 to 4 coping strategies written in full sentences — these can include grounding techniques, journaling, exercise, mindfulness, or reaching out to someone.
3. End your message with a kind and encouraging tone.

Do not diagnose or offer clinical advice.

If the user seems in significant distress, suggest seeking professional help. You may recommend:
“You can contact Samaritans at 116 123 or visit their website for free, 24/7 support.” or “You can contact Mind at 0300 102 1234 or visit their website for free, 24/7 support.”

Do not repeat the welcome message once the user has already started sharing. Only show that message at the beginning of a new chat.
If the user expresses feeling overwhelmed, exhausted, or emotionally burdened, do not continue probing indefinitely. Provide supportive feedback and offer coping mechanisms.

Always speak with warmth, safety, and care. Your job is to make the user feel heard, understood, and supported.
"""

# Home route
@app.route('/')
def home():
    if 'messages' not in session:
        session['messages'] = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Hello! How are you feeling today? What would you like to talk about?"}
        ]
        session['emotional_count'] = 0
    return render_template('index.html')

# Helper function to check if a message is emotionally expressive
def is_emotionally_detailed(text):
    keywords = ["anxious", "depressed", "breaking point", "struggling", "overwhelmed", "i can't take", "i can't cope", "crying", "can't sleep", "panic", "hopeless"]
    return any(word in text.lower() for word in keywords)

# Chat route
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message received.'}), 400

    if 'messages' not in session:
        session['messages'] = [{"role": "system", "content": system_prompt}]
        session['emotional_count'] = 0

    session['messages'].append({"role": "user", "content": user_message})

    # Count emotionally expressive responses
    if is_emotionally_detailed(user_message):
        session['emotional_count'] += 1

    # If enough emotional depth, append signal to shift tone
    if session['emotional_count'] >= 2:
        session['messages'].append({
            "role": "system",
            "content": "The user has expressed significant emotional detail. Please shift to providing support and coping strategies now."
        })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=session['messages'],
            temperature=0.4,
            max_tokens=800
        )

        assistant_message = response.choices[0].message.content.strip()
        session['messages'].append({"role": "assistant", "content": assistant_message})

        return jsonify({'response': assistant_message})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


#Run app
if __name__ == '__main__':
    app.run(debug=True)
