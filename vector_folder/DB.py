import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
load_dotenv()
from langchain_core.documents import Document
docs = [
    Document(page_content ="Python: A programming language used for data analysis, automation, machine learning, and application development.",metadata = {"source": "AI_book"}),
Document(page_content ="Pandas: A Python library used for data manipulation, cleaning, and analysis.",metadata={"source": "DataScience"}),
Document(page_content ="Neural Network: A machine learning model inspired by the human brain that learns patterns from data using interconnected layers of neurons.",metadata = {"source":"DL_book"})
]
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma-db"
)
result = vectorstore.similarity_search("what is used for data analysis?",k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)
retriver = vectorstore.as_retriever()
docs = retriver.invoke("explain deeo learning")

for d in docs:
    print(d.page_content)
    