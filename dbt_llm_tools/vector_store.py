import os
import json
import boto3
import chromadb

from dbt_llm_tools.dbt_model import DbtModel
from dbt_llm_tools.types import ParsedSearchResult


class VectorStore:
    def __init__(
            self,
            bedrock_model_id: str = "amazon.titan-embed-text-v1",
            vector_db_path: str = ".local_storage/chroma.db",
            test_mode: bool = False,
    ) -> None:
        if not isinstance(vector_db_path, str) or vector_db_path == "":
            raise Exception("Please provide a valid path for the persistent database.")

        os.makedirs(vector_db_path, exist_ok=True)
        self.__client = chromadb.PersistentClient(vector_db_path)
        self.__collection_name = "model_documentation"

        self.__bedrock_client = boto3.client(service_name="bedrock-runtime")
        self.__bedrock_model_id = bedrock_model_id
        self.__test_mode = test_mode

        self.__collection = self.__create_collection()

    def __embed_text(self, text: str):
        if self.__test_mode:
            return [0.0] * 1536  # Dummy vector for testing

        response = self.__bedrock_client.invoke_model(
            modelId=self.__bedrock_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    def __create_collection(self, distance_fn: str = "l2") -> chromadb.Collection:
        return self.__client.get_or_create_collection(
            name=self.__collection_name,
            metadata={"hnsw:space": distance_fn},
        )

    def get_client(self) -> chromadb.PersistentClient:
        return self.__client

    def upsert_models(self, models: list[DbtModel]) -> None:
        documents = []
        metadatas = []
        ids = []

        for model in models:
            if not isinstance(model, DbtModel):
                raise Exception("Please provide a list of valid dbt model objects.")

            model_text = model.as_prompt_text()
            embedding = self.__embed_text(model_text)

            documents.append(model_text)
            metadatas.append({"tags": json.dumps(model.tags), "embedding": embedding})
            ids.append(model.name)

        return self.__collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    def get_models(self, model_ids: list[str] = None) -> list[DbtModel]:
        models = []
        raw_models = self.__collection.get(ids=model_ids)

        for i in range(len(raw_models["ids"])):
            models.append({"id": raw_models["ids"][i], "document": raw_models["documents"][i]})

        return models

    def query_collection(self, query: str, n_results: int = 3) -> list[ParsedSearchResult]:
        if not isinstance(query, str) or query == "":
            raise Exception("Please provide a valid query.")

        query_embedding = self.__embed_text(query)

        search_results = self.__collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )

        closest_models = []
        for i in range(len(search_results["ids"][0])):
            closest_models.append(
                {
                    "id": search_results["ids"][0][i],
                    "metadata": search_results["metadatas"][0][i],
                    "document": search_results["documents"][0][i],
                    "distance": search_results["distances"][0][i],
                }
            )

        return closest_models

    def reset_collection(self) -> None:
        self.__client.delete_collection(self.__collection_name)
        self.__collection = self.__create_collection()