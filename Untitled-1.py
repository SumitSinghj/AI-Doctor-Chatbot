# %%
# importing Libraries
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# %%
# Download NLTK Resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# %%
# %% Load and preprocess data
with open('data.txt', 'r', errors='ignore') as f:
    raw_doc = f.read().lower()

# Split dataset into blocks
entries = raw_doc.strip().split('\n\n')

# Extract all lines from blocks for TF-IDF matching
sentence_token = []
for entry in entries:
    lines = entry.split('\n')
    for line in lines:
        if line.startswith("symptoms:"):
            sentence_token.append(line.strip())


# %%
#Define Text Normalization Functions
lemmer = nltk.stem.WordNetLemmatizer()
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


# %%
# Greeting Detection logic

greet_inputs= ("hello", 'hi', 'whatsup', 'how are you?')
greet_response= ('hi', 'Hey', 'Hey There!','How can I help you?')
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_response)

# %%
def response(user_response):
    sentence_token.append(user_response)
    tfidf = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english').fit_transform(sentence_token)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    sentence_token.pop()

    # Find full entry containing that matched symptom line
    matched_symptom = sentence_token[idx]
    for entry in entries:
        if matched_symptom in entry:
            return entry.strip()

    return "I'm sorry, I couldn't identify your condition. Please rephrase or consult a doctor."

# %%
print("AI_Doctor Bot 🤖: Hello! I'm here to help you understand your symptoms. Type 'bye' to exit.")

while True:
    user_response = input("You 💬: ").lower()
    if user_response in ['bye', 'exit', 'quit']:
        print("AI_Doctor Bot 🤖: Take care! See you next time.")
        break
    elif user_response in ['thank you', 'thanks', 'thankyou']:
        print("AI_Doctor Bot 🤖: You're very welcome!")
        break
    elif greet(user_response):
        print("AI_Doctor Bot 🤖:", greet(user_response))
    else:
        print("AI_Doctor Bot 🤖:", response(user_response))


# %%



