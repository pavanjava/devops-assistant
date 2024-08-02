from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
    draw_most_recent_execution
)

from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
import qdrant_client


class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]


class RAGWorkflow(Workflow):

    def __init__(self, timeout: int = 30, verbose: bool = False):
        super().__init__(timeout, verbose)
        text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)
        client = qdrant_client.QdrantClient(url="http://localhost:6333/", api_key="th3s3cr3tk3y")
        self.vector_store = QdrantVectorStore(client=client, collection_name="iac")
        Settings.transformations = [text_parser]

    @step(pass_context=True)
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        documents = SimpleDirectoryReader(input_files=['main.tf']).load_data()
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        ctx.data["index"] = VectorStoreIndex.from_documents(documents=documents,
                                                            storage_context=storage_context,
                                                            transformations=Settings.transformations,
                                                            embed_model=OllamaEmbedding(
                                                                model_name='mxbai-embed-large:latest',
                                                                base_url='http://localhost:11434'),
                                                            show_progress=True)

        return StopEvent(result=f"Indexed {len(documents)} documents.")

    @step(pass_context=True)
    async def retrieve(
            self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        ctx.data["query"] = query

        # get the index from the global context
        index = ctx.data.get("index")
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step(pass_context=True)
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        llm = Ollama(model='llama3.1', base_url='http://localhost:11434')
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = ctx.data.get("query")

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)


async def run_rag():
    w = RAGWorkflow(timeout=60, verbose=True)
    # Ingest the documents
    await w.run(dirname="data")

    # Run a query
    result = await w.run(query="what is the storage account name given in terraform script ?")
    async for chunk in result.async_response_gen():
        print(chunk, end="", flush=True)

    draw_most_recent_execution(w, filename='flow.html')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main=run_rag())
