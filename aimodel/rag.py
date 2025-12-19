import bs4
from langsmith import Client
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from baseLLM import VLLM


loader = WebBaseLoader(
    web_paths= ("https://lilianweng.github.io/posts/2023-06-23-agent/","https://medium.com/@KonstantinPM/methods-and-examples-of-task-decomposition-in-product-development-ed578816e4cc","https://en.wikipedia.org/wiki/Temporary_file"),
    bs_kwargs= dict(
        parse_only = bs4.SoupStrainer(
            class_ = ("post-content", "post-title", "post-header",
                      "pw-post-body-paragraph nl nm hl nn b no np nq nr ns nt nu nv nw nx ny nz oa ob oc od oe of og oh oi he bl",
                      "mw-content-ltr mw-parser-output")
        )
    ),
)
docs = loader.load()

model = SentenceTransformer("all-MiniLM-L6-v2")

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()


model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding = SentenceTransformerEmbeddings(model_name)

textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
splits = textsplitter.split_documents(docs)

vectorStore = Chroma.from_documents(
    documents=splits,
    collection_name="dcd_store",
    embedding=embedding)

retriever = vectorStore.as_retriever()

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt") 

from langchain_openai import ChatOpenAI

llm = VLLM(base_url="http://0.0.0.0:16520", api_key="expertKey")
output = llm("What is the capital of France?")
print(output)


response = llm.invoke("What is Task Decomposition?")
print(response)
