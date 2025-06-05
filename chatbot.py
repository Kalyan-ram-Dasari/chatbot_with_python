import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download punkt tokenizer if not already downloaded
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

# Load intents file (use raw string for Windows paths)
with open(r'C:\Users\Lenovo\Documents\chatbot\intents.json', encoding='utf-8') as f:
    intents = json.load(f)

# Load preprocessed words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load trained model
model = load_model('chatbot_model.h5')

# Clean sentence into tokens
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into bag of words array
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Your exact predict_class function (no changes)
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if not results:
        # No intent over threshold, return unknown intent
        return [{'intent': 'unknown', 'probability': '1.0'}]
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Get a random response from predicted intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you try again?"

    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Chat loop
print("GO! Bot is running!")

while True:
    message = input("You: ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)
    
