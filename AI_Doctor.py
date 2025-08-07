import streamlit as st
import pickle
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

# Load data
with open("symptom_sentences.pkl", "rb") as f:
    sentence_token = pickle.load(f)

with open("chatbot_entries.pkl", "rb") as f:
    entries = pickle.load(f)

# Preprocessing
lemmer = nltk.stem.WordNetLemmatizer()
remove_punc_dict = dict((ord(p), None) for p in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(t) for t in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

def get_response(user_input):
    temp_token = sentence_token + [user_input]
    tfidf = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english').fit_transform(temp_token)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = vals.argsort()[0][-1]
    matched = sentence_token[idx]
    for entry in entries:
        if matched in entry:
            return f"""
            Based on your symptoms, here's what I found:
            
            **{entry.strip()}**
            
            *This information is for educational purposes only.*
            """
    return "🤷‍♂️ I couldn't find a precise match for your symptoms. Could you provide more details about:\n- Duration\n- Severity\n- Any other associated symptoms?"

# Page setup
st.set_page_config(page_title="AI Doctor Chatbot", page_icon="🩺", layout="wide")
st.title("🩺 AI Doctor Chatbot")
st.markdown("Describe your symptoms and I'll try to help. Type 'bye' to end the conversation.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Common symptoms buttons
st.write("Common symptoms:")
cols = st.columns(4)
with cols[0]:
    if st.button("Fever", use_container_width=True, key="fever_btn"):
        user_msg = "I have fever"
        st.session_state.messages.append({"role": "user", "content": user_msg})
        # Generate and add bot response
        bot_response = get_response(user_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()

with cols[1]:
    if st.button("Headache", use_container_width=True, key="headache_btn"):
        user_msg = "I have headache"
        st.session_state.messages.append({"role": "user", "content": user_msg})
        bot_response = get_response(user_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()

with cols[2]:
    if st.button("Cough", use_container_width=True, key="cough_btn"):
        user_msg = "I have persistent cough"
        st.session_state.messages.append({"role": "user", "content": user_msg})
        bot_response = get_response(user_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()

with cols[3]:
    if st.button("Fatigue", use_container_width=True, key="fatigue_btn"):
        user_msg = "I feel constant fatigue"
        st.session_state.messages.append({"role": "user", "content": user_msg})
        bot_response = get_response(user_msg)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.rerun()

# Accept user input
if prompt := st.chat_input("Describe your symptoms here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    if prompt.lower() == "bye":
        response = "Goodbye! Feel free to come back if you have more health concerns."
    else:
        response = get_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})