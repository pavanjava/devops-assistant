from llama_index.core.workflow import (
    draw_most_recent_execution
)
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from infrastructure_as_code_assistant import InfrastructureAsCodeAssistant
from rag_on_iaac import RAGWorkflow
import logging

logging.basicConfig(level=logging.INFO)
Settings.embed_model = OllamaEmbedding(model_name='mxbai-embed-large:latest', base_url='http://localhost:11434')


async def main():
    w = InfrastructureAsCodeAssistant(timeout=60, verbose=True)
    result = await w.run(topic="Azure Storage Account")
    print(str(result))
    w = RAGWorkflow(timeout=60, verbose=True)
    # Ingest the documents
    await w.run(dirname="data")

    # Run a query
    result = await w.run(query="what is the storage account name ?")
    async for chunk in result.async_response_gen():
        print(chunk, end="", flush=True)
    # draw_most_recent_execution(w, filename='flow.html')


if __name__ == '__main__':
    import asyncio
    asyncio.run(main=main())
