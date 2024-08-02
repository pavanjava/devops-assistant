from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step
)
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import logging

logging.basicConfig(level=logging.INFO)
Settings.embed_model = OllamaEmbedding(model_name='mxbai-embed-large:latest', base_url='http://localhost:11434')


class TerraformScriptEvent(Event):
    script: str


class ValidationErrorEvent(Event):
    error: str
    wrong_output: str
    passage: str


class InfrastructureAsCodeAssistant(Workflow):
    llm = Ollama(model='llama3.1', base_url='http://localhost:11434', temperature=0.8, request_timeout=300)

    @step()
    async def create_script(self, ev: StartEvent) -> TerraformScriptEvent:
        try:
            topic = ev.get("topic")
            prompt = f"Write optimized terraform script for {topic}. No explanation is needed just give the script only."
            logging.info(f'create_script_prompt: {prompt}')
            response = await self.llm.acomplete(prompt)
            logging.info(f'generated script: {response}')
            return TerraformScriptEvent(script=str(response))
        except Exception as e:
            print("Creation failed,...")
            logging.error(str(e))

    @step()
    async def validate_script(self, ev: TerraformScriptEvent) -> TerraformScriptEvent:
        try:
            terraform_script = ev.script
            prompt = (f"Assume you are a senior devops engineer and given the terraform script: {terraform_script}, "
                      f"your job is to validate the script before user execute it in order to reduce the error time."
                      f"don't give any comments if no deviations found the script. comment if only script has errors")
            logging.info(f'validating the script: {prompt}')
            response = await self.llm.acomplete(prompt)
            logging.info(f'after validation: {response}')
            return TerraformScriptEvent(script=str(response))
        except Exception as e:
            print("Validation failed, ...")
            logging.error(str(e))

    @step()
    async def save_script(self, ev: TerraformScriptEvent) -> StopEvent:
        try:
            terraform_script = ev.script
            with open('main.tf', mode='w') as script:
                script.write(terraform_script)
            return StopEvent(result=str(terraform_script))
        except Exception as e:
            print("Script writing failed, ...")
            logging.error(str(e))
