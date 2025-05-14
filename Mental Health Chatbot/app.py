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

#Backup system prompt for system behviour
system_prompt = """
You are a compassionate mental health assistant chatbot.

Your role is to support users through difficult emotions such as anxiety, depression, stress, grief, or family issues. You are not a therapist or a doctor. Your job is to listen, reflect, and provide gentle, helpful coping suggestions.

At the start of a conversation:
- Greet the user warmly. Example: "Hello! How are you feeling today? What would you like to talk about?"

During the conversation:
- When the user expresses distress, validate their feelings.
- Ask one gentle, open-ended question to encourage deeper sharing.

**Important:** If the user shares emotionally detailed experiences in two or more consecutive messages (or seems overwhelmed), stop asking further questions.
- Reflect empathetically.
-Offer 3 to 4 coping strategies in full sentences.
    Start each strategy on a new line, using this format:
        1. **Strategy title**: Explanation.
        
    For example:
        1. **Practice Mindfulness**: Explanation here.
        2. **Physical Activity**: Explanation here.
- End with encouragement and remind them they're not alone.
- If the user's message is unrelated to mental health (e.g., asking for jokes, general knowledge, math problems, technology help, etc.), politely remind them: 
  'I'm here to help with mental health support. Please send a message related to your feelings, emotions, stress, anxiety, or mental wellbeing.'
- If the user sends 3 consecutive irrelevant messages, reset the conversation by saying:
  'I'm only able to continue if we focus on mental health topics. Let's start fresh. How are you feeling today?'

If at any point the user expresses thoughts of self-harm or suicide, respond with:
"I'm really concerned about your safety. It's important to talk to someone immediately. You can call Samaritans at 116 123 or visit their website for free, 24/7 support."

Do not offer diagnoses or medical advice.
Always use a warm, caring tone.
"""

def get_custom_prompt(user_message):
    message = user_message.lower()

    if "family" in message:
        return "The user is describing distress related to family conflict. Ask an open-ended question specifically about their family situation."
    elif "school" in message or "teacher" in message or "homework" in message:
        return "The user is discussing school stress. Ask what about school is causing stress and guide them with strategies to cope in that environment."
    elif "lonely" in message or "alone" in message:
        return "The user feels lonely. Ask about their support network or if they've been feeling disconnected lately."
    elif "relationship" in message or "partner" in message:
        return "The user is describing relationship problems. Ask what's been difficult in their relationship lately."
    elif "angry" in message or "lash out" in message:
        return "The user is describing anger or frustration. Ask what has been triggering these feelings and how often they occur."
    elif "sober" in message or "alcohol" in message:
        return "The user is in recovery and feeling unstable. Ask how they're coping with sobriety and offer emotional validation."
    
    return ""


#Home route
@app.route('/')
def home():
    if 'messages' not in session:
        session['messages'] = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Hello! How are you feeling today? What would you like to talk about?"}
        ]
        session['emotional_count'] = 0
    return render_template('index.html')

#Helper function to check if a message is emotionally expressive
def is_emotionally_detailed(text):
    keywords = ["anxious", "depressed", "breaking point", "struggling", "overwhelmed", "i can't take", "i can't cope", "crying", "can't sleep", "panic", "hopeless"]
    return any(word in text.lower() for word in keywords)

#Chat route
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')

    #Special handling for welcome message
    if user_message == "__WELCOME__":
        session['messages'] = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Hello! How are you feeling today? What would you like to talk about?"}
        ]
        session['emotional_count'] = 0
        return jsonify({'response': "Hello! How are you feeling today? What would you like to talk about?"})

    if not user_message:
        return jsonify({'error': 'No message received.'}), 400

    if 'messages' not in session:
        session['messages'] = [{"role": "system", "content": system_prompt}]
        session['emotional_count'] = 0

    session['messages'].append({"role": "user", "content": user_message})

    #Count emotionally expressive responses
    if is_emotionally_detailed(user_message):
        session['emotional_count'] += 1

    #If enough emotional depth, append signal to shift tone
    if session['emotional_count'] >= 2:
        session['messages'].append({
            "role": "system",
            "content": "The user has expressed significant emotional detail. Please shift to providing support and coping strategies now."
        })

    #Inject targeted system prompt
    targeted_prompt = get_custom_prompt(user_message)
    if targeted_prompt:
        session['messages'].append({
            "role": "system",
            "content": targeted_prompt
        })


    try:
        response = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:ethans-honours-project:mental-health-v2:BW5ntG3O",
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
