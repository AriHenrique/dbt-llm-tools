import json
import os

import yaml
import boto3

from dbt_llm_tools.dbt_project import DbtProject
from dbt_llm_tools.instructions import ANSWER_QUESTION_INSTRUCTIONS
from dbt_llm_tools.types import ParsedSearchResult, PromptMessage
from dbt_llm_tools.vector_store import VectorStore


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


class Chatbot:
    def __init__(
            self,
            dbt_project_root: str,
            bedrock_model_id: str = "anthropic.claude-v2",
            database_path: str = ".local_storage/db.json",
            vector_db_path: str = ".local_storage/chroma.db",
    ) -> None:
        self.__bedrock_model_id: str = bedrock_model_id

        self.project: DbtProject = DbtProject(
            dbt_project_root=dbt_project_root, database_path=database_path
        )

        self.store: VectorStore = VectorStore(vector_db_path=vector_db_path)
        self.__bedrock_client = boto3.client(service_name="bedrock-runtime")
        self.__instructions: list[str] = [ANSWER_QUESTION_INSTRUCTIONS]

    def __prepare_prompt(
            self, closest_models: list[ParsedSearchResult], query: str
    ) -> list[PromptMessage]:
        prompt: list[PromptMessage] = []

        for instruction in self.__instructions:
            prompt.append({"role": "system", "content": instruction})

        for model in closest_models:
            prompt.append({"role": "system", "content": model["document"]})

        prompt.append({"role": "user", "content": query})

        return prompt

    def get_instructions(self) -> list[str]:
        return self.__instructions

    def set_instructions(self, instructions: list[str]) -> None:
        self.__instructions = instructions

    def load_models(
            self,
            models: list[str] = None,
            included_folders: list[str] = None,
            excluded_folders: list[str] = None,
    ) -> None:
        models = self.project.get_models(models, included_folders, excluded_folders)
        self.store.upsert_models(models)

    def reset_model_db(self) -> None:
        self.store.reset_collection()

    def ask_question(self, query: str, get_model_names_only: bool = False) -> str:
        print("Asking question: ", query)

        print("\nLooking for closest models to the query...")
        closest_models = self.store.query_collection(query)
        model_names = ", ".join(map(lambda x: x["id"], closest_models))

        if get_model_names_only:
            return model_names

        print("Closest models found:", model_names)
        print("\nPreparing prompt...")
        prompt = self.__prepare_prompt(closest_models, query)

        print("\nCalculating response...")
        response = self.__bedrock_client.invoke_model(
            modelId=self.__bedrock_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"input": prompt}),
        )

        response_body = json.loads(response["body"].read())
        print("\nResponse received: \n")
        print(response_body["completion"])

        return response_body["completion"]
