import streamlit as st
from tinydb import TinyDB, Query

def load_session_state_from_db():
    if "local_db_path" not in st.session_state:
        st.session_state["local_db_path"] = ".local_storage/db.json"

    db = TinyDB(st.session_state.get("local_db_path"))
    settings = db.get(Query().type == "settings")

    if settings is not None:
        for key in settings:
            if key != "type":
                st.session_state[key] = settings[key]

    if "bedrock_chatbot_model" not in st.session_state:
        st.session_state["bedrock_chatbot_model"] = "anthropic.claude-v2"

    if "bedrock_embedding_model" not in st.session_state:
        st.session_state["bedrock_embedding_model"] = "amazon.titan-embed-text-v1"

    if "vector_store_path" not in st.session_state:
        st.session_state["vector_store_path"] = ".local_storage/chroma.db"

def save_session_to_db():
    db = TinyDB(
        st.session_state.get("local_db_path", ".local_storage/db.json"),
        sort_keys=True,
        indent=4,
    )
    db.upsert(
        {
            "type": "settings",
            "dbt_project_root": st.session_state.get("dbt_project_root", ""),
            "bedrock_chatbot_model": st.session_state.get("bedrock_chatbot_model", ""),
            "bedrock_embedding_model": st.session_state.get("bedrock_embedding_model", ""),
            "vector_store_path": st.session_state.get("vector_store_path", ".local_storage"),
            "local_db_path": st.session_state.get("local_db_path", ".local_storage/db.json"),
        },
        Query().type == "settings",
        )

    st.toast("Settings saved to file!", icon="📁")