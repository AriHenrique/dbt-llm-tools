import streamlit as st
import boto3
from tinydb import TinyDB, Query

from menu import menu
from settings import load_session_state_from_db
from dbt_llm_tools import VectorStore
from dbt_llm_tools.instructions import ANSWER_QUESTION_INSTRUCTIONS

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

menu()
load_session_state_from_db()

st.session_state.is_new_question = len(st.session_state.get("messages", [])) == 0

vector_store = VectorStore(
    vector_db_path=st.session_state.get("vector_store_path", ".local_storage/chroma.db"),
)

bedrock_client = boto3.client(service_name="bedrock-runtime")

def get_matching_models(query):
    return vector_store.query_collection(query=query, n_results=4)

st.title("Question Answerer")
st.text("Ask a chatbot questions about your data!")

# Set a default model
if "bedrock_chatbot_model" not in st.session_state:
    st.session_state["bedrock_chatbot_model"] = "anthropic.claude-v2"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

if prompt := st.chat_input("What is up?"):
    if st.session_state.is_new_question:
        st.session_state.closest_model_names = []
        st.session_state.messages += [
            {"role": "system", "content": ANSWER_QUESTION_INSTRUCTIONS},
            {"role": "system", "content": "The user would like to know:"},
        ]
        st.session_state.is_new_question = False

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    closest_models = get_matching_models(prompt)

    if closest_models:
        st.session_state.messages.append(
            {
                "role": "system",
                "content": """
                    In addition to information you have already, here is more information about certain tables
                    that might help you answer the user's question:
                """,
            }
        )

        for model in closest_models:
            if model["id"] not in st.session_state.closest_model_names:
                st.session_state.messages.append(
                    {"role": "system", "content": model["document"]}
                )
                st.session_state.closest_model_names.append(model["id"])

    with st.chat_message("assistant"):
        response = bedrock_client.invoke_model(
            modelId=st.session_state["bedrock_chatbot_model"],
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"input": st.session_state.messages}),
        )
        response_body = json.loads(response["body"].read())
        st.write(response_body["completion"])

    st.session_state.messages.append({"role": "assistant", "content": response_body["completion"]})


def clear_chat():
    st.session_state.is_new_question = True
    st.session_state.messages = []
    st.session_state.closest_model_names = []
    st.toast("Starting over!")

if st.session_state.is_new_question is False:
    st.button("Start over", on_click=clear_chat)