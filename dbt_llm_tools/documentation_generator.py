import json
import os

import yaml
import boto3

from dbt_llm_tools.dbt_project import DbtProject
from dbt_llm_tools.instructions import INTERPRET_MODEL_INSTRUCTIONS
from dbt_llm_tools.types import DbtModelDict, DbtModelDirectoryEntry, PromptMessage


class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


class DocumentationGenerator:
    def __init__(
            self,
            dbt_project_root: str,
            bedrock_model_id: str = "anthropic.claude-v2",
            database_path: str = "./directory.json",
    ) -> None:
        self.dbt_project = DbtProject(
            dbt_project_root=dbt_project_root, database_path=database_path
        )

        self.__bedrock_client = boto3.client(service_name="bedrock-runtime")
        self.__bedrock_model_id = bedrock_model_id

    def __get_system_prompt(self, message: str) -> PromptMessage:
        return {
            "role": "system",
            "content": message,
        }

    def __save_interpretation_to_yaml(
            self, model: DbtModelDict, overwrite_existing: bool = False
    ) -> None:
        yaml_path = model.get("yaml_path")

        if yaml_path is not None:
            if not overwrite_existing:
                raise Exception(
                    f"Model already has documentation at {model['yaml_path']}"
                )

            with open(model["yaml_path"], "r", encoding="utf-8") as infile:
                existing_yaml = yaml.load(infile, Loader=yaml.FullLoader)
                existing_models = existing_yaml.get("models", [])

                search_idx = -1
                for idx, m in enumerate(existing_models):
                    if m["name"] == model["name"]:
                        search_idx = idx

                if search_idx != -1:
                    existing_models[search_idx] = model["interpretation"]
                else:
                    existing_models.append(model["interpretation"])

                existing_yaml["models"] = existing_models
                yaml_content = existing_yaml
        else:
            model_path = model["absolute_path"]
            head, tail = os.path.split(model_path)
            yaml_path = os.path.join(head, "_" + tail.replace(".sql", ".yml"))

            yaml_content = {"version": 2, "models": [model["interpretation"]]}

        with open(yaml_path, "w", encoding="utf-8") as outfile:
            yaml.dump(
                yaml_content,
                outfile,
                Dumper=MyDumper,
                default_flow_style=False,
                sort_keys=False,
            )

    def interpret_model(self, model: DbtModelDirectoryEntry) -> DbtModelDict:
        print(f"Interpreting model: {model['name']}")

        prompt = []
        refs = model.get("refs", [])

        prompt.append(self.__get_system_prompt(INTERPRET_MODEL_INSTRUCTIONS))
        prompt.append(
            self.__get_system_prompt(
                f"""
                The model you are interpreting is called {model["name"]} following is the Jinja SQL code for the model:
                {model.get("sql_contents")}
                """
            )
        )

        if len(refs) > 0:
            prompt.append(
                self.__get_system_prompt(
                    f"""
                    The model {model["name"]} references the following models: {", ".join(refs)}.
                    The interpretation for each of these models is as follows:
                    """
                )
            )

            for ref in refs:
                ref_model = self.dbt_project.get_single_model(ref)
                prompt.append(
                    self.__get_system_prompt(
                        f"""
                        The model {ref} is interpreted as follows:
                        {json.dumps(ref_model.get("interpretation"), indent=4)}
                        """
                    )
                )

        response = self.__bedrock_client.invoke_model(
            modelId=self.__bedrock_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"input": prompt}),
        )
        response_body = json.loads(response["body"].read())
        return json.loads(response_body["completion"])

    def generate_documentation(
            self, model_name: str, write_documentation_to_yaml: bool = False
    ) -> DbtModelDict:
        model = self.dbt_project.get_single_model(model_name)

        for dep in model.get("deps", []):
            dep_model = self.dbt_project.get_single_model(dep)
            if dep_model.get("interpretation") is None:
                dep_model["interpretation"] = self.interpret_model(dep_model)
                self.dbt_project.update_model_directory(dep_model)

        interpretation = self.interpret_model(model)
        model["interpretation"] = interpretation

        if write_documentation_to_yaml:
            self.__save_interpretation_to_yaml(model)

        self.dbt_project.update_model_directory(model)
        return interpretation
