import json
import os

import streamlit as st
import boto3
from tinydb import TinyDB, Query
from tinydb.operations import set

from menu import menu
from styles import button_override
from settings import load_session_state_from_db, save_session_to_db

from dbt_llm_tools import DbtProject

st.set_page_config(page_title="dbt-llm-tools", page_icon="🤖", layout="wide")

db = TinyDB(st.session_state.get("local_db_path", ".local_storage/db.json"))

st.title("Welcome to dbt-llm-tools 👋")

st.caption(
    "dbt-llm-tools is brought to you by [Pragun Bhutani](https://pragunbhutani.com/)."
)

st.subheader("Get started")
st.text(
    "To get started, choose where you would like to save your project settings and click on 'Get Started'."
)

menu()
button_override()
load_session_state_from_db()

if bedrock_model_id := st.text_input(
        label="AWS Bedrock Model ID",
        help="Enter the AWS Bedrock Model ID",
        value=st.session_state.get("bedrock_model_id", "anthropic.claude-v2"),
):
    st.session_state["bedrock_model_id"] = bedrock_model_id
    db.update(set("bedrock_model_id", bedrock_model_id), Query().type == "settings")

if dbt_project_root := st.text_input(
        label="DBT Project Root",
        help="Path to the folder that contains your dbt_project.yml file.",
        value=st.session_state.get("dbt_project_root", ""),
):
    st.session_state["dbt_project_root"] = dbt_project_root
    db.update(set("dbt_project_root", dbt_project_root), Query().type == "settings")

st.caption("")

if st.button(
        label="Parse Project",
        type="primary",
        help="Parse the DBT project. Project Root must be tested first.",
        disabled=not st.session_state.get("dbt_project_root", False),
):
    dbt_project = DbtProject(
        dbt_project_root=st.session_state["dbt_project_root"],
        database_path=st.session_state["local_db_path"],
    )
    dbt_project.parse()
    save_session_to_db()
    st.success("Project Parsed Successfully!")

st.divider()

st.subheader("Additional Settings")
st.text("Select your AWS Bedrock language model and embedding model.")

if bedrock_chatbot_model := st.selectbox(
        "Chatbot Model",
        ("anthropic.claude-v2", "anthropic.claude-v1", "amazon.titan-text-lite-v1"),
        help="The model you select will be used to generate responses for your chatbot.",
):
    st.session_state["bedrock_chatbot_model"] = bedrock_chatbot_model
    db.update(
        set("bedrock_chatbot_model", bedrock_chatbot_model), Query().type == "settings"
    )

if bedrock_embedding_model := st.selectbox(
        "Embedding Model",
        ("amazon.titan-embed-text-v1"),
        help="The model you select will be used to generate embeddings for your chatbot.",
):
    st.session_state["bedrock_embedding_model"] = bedrock_embedding_model
    db.update(
        set("bedrock_embedding_model", bedrock_embedding_model),
        Query().type == "settings",
        )

st.caption("")

st.divider()

st.subheader("Dangerous actions")
st.text("Choose where you would like to store your project data.")

st.caption("")

if st.button(
        "Reset local storage",
        disabled="local_db_path" not in st.session_state,
        type="primary",
):
    db = TinyDB(st.session_state["local_db_path"])
    File = Query()
    db.remove(File.type == "settings")
    db.remove(File.type == "model")
    db.remove(File.type == "source")

    st.toast("Settings cleared from file!", icon="📁")

st.divider()
