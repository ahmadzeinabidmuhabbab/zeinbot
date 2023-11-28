# import library
import streamlit as st
# import random
import time
# import logging
# import json
import pickle
import string
import numpy as np
from util import JSONParser
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs

knowledge_base = open("knowledge_base.txt", "r").read().replace('\n', '')
# Configure the model
@st.cache_resource
def modelQA():
    model_args = QuestionAnsweringArgs()
    QA_model = QuestionAnsweringModel(
        "bert", "cahya/bert-base-indonesian-tydiqa", args=model_args, use_cuda=False
    )
    return QA_model
    
def preprocess(chat):
    # konversi ke non kapital
    chat = chat.lower()
    # hilangkan tanda baca
    tandabaca = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tandabaca)
    return chat

def bot_response(chat, pipeline, jp):
    chat = preprocess(chat)
    res = pipeline.predict_proba([chat])
    max_prob = max(res[0])
    if max_prob < 0.2:
        # QA_model = pickle.load(open("QA_model.pkl", 'rb'))
        
        # Make predictions with the model
        to_predict = [
            [
                {
                    "context": knowledge_base,
                    "qas": [
                        {
                            "question": chat,
                            "id": "0",
                        }
                    ],
                }
            ]
        ]
        answers, probabilities = modelQA().predict(to_predict[0])
        answer_fix = ''
        for i in range(0,len(answers[0]['answer'])):
            if answers[0]['answer'][i] != '':
                answer_fix = answers[0]['answer'][i]
                break
        return answer_fix , None
    else:
        max_id = np.argmax(res[0])
        pred_tag = pipeline.classes_[max_id]
        return jp.get_response(pred_tag), pred_tag

# load data
path = "data/intents.json"
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

# praproses data
# case folding -> transform kapital ke non kapital, hilangkan tanda baca
df['text_input_prep'] = df.text_input.apply(preprocess)

# Load model QA Scikit
@st.cache_resource
def pipeline():
    pipeline_model = pickle.load(open("model_chatbot.pkl", 'rb'))
    return pipeline_model

st.title("ZeinBot: Tanya\" Santuy")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Mau tanya apa?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display assistant response in chat message container
    with st.chat_message("assitant"):
        message_placeholder = st.empty()
        full_response = ""
        chat = prompt
        res, tag = bot_response(chat, pipeline(), jp)
        # assistant_response = random.choice(
        #     [
        #         "Hello there! How can I assist you today?",
        #         "Hi, human! Is there anything I can help you with?",
        #         "Do you need help?",
        #     ]
        # )
        assistant_response = res
        
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


