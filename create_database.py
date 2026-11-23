from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
load_dotenv()
data = PyPDFLoader("lbdl.pdf")
docs = data.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
        chunk_overlap=200)
chunks= splitter.split_documents(docs)
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
    )
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = "chroma-db"
)
result = vectorstore.similarity_search("what is used for data analysis?",k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)
retriver = vectorstore.as_retriever()
docs = retriver.invoke("explain deep learning")

for d in docs:
    print(d.page_content)
                     