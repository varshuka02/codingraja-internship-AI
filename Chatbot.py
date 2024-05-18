import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import spacy
import tensorflow as tf

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    tokens = word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def recognize_intent(text):
    doc = nlp(text)
    intent = None
    for token in doc:
        if token.text.lower() in ['hello', 'hi', 'hey']:
            intent = 'greeting'
        elif token.text.lower() in ['goodbye', 'bye', 'see', 'later']:
            intent = 'farewell'
    return intent or 'unknown'

def generate_response(intent):
    if intent == 'greeting':
        return random.choice(['Hello!', 'Hi there!', 'Hey! How can I help you?'])
    elif intent == 'farewell':
        return random.choice(['Goodbye!', 'See you later!', 'Bye! Have a great day!'])
    else:
        return "I'm sorry, I didn't understand that."

print("Chatbot: Hello! How can I assist you today?")

def collect_data(user_input, intent):
    with open("conversation_dataset.txt", "a") as file:
        file.write(f"User Input: {user_input}\tIntent: {intent}\n")


Q = {}
alpha = 0.1

gamma = 0.9

def update_Q_value(state, action, reward, next_state):
    current_Q_value = Q.get((state, action), 0.0)
    next_max = max(Q.get((next_state, next_action), 0.0) for next_action in ['greeting', 'farewell', 'unknown'])
    new_Q_value = current_Q_value + alpha * (reward + gamma * next_max - current_Q_value)
    Q[(state, action)] = new_Q_value

while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye! Have a great day!")
        break
    
    
    processed_input = preprocess(user_input)
    
   
    intent = recognize_intent(user_input)
    
   
    collect_data(user_input, intent)
    
    
    response = generate_response(intent)
    
    
    print("Chatbot:", response)

    
    update_Q_value(user_input, intent, 1, processed_input) 


