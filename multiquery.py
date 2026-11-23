from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

docs =[Document(page_content="gardient docuemnt is an optimizatino algorithm used "),
       Document(page_content="gardient is minimize the loss "),
       Document(page_content= "gradient descent is organise and minimizes the loss"),
       Document(page_content= "Neural network use gradient decent for training."),
       Document(page_content = "support vector machine are supervised leaning algorithm")
]
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectorstore = Chroma.from_documents(docs,embeddings)
retriever = vectorstore.as_retriever()
llm = ChatMistralAI(model = "mistral-small-latest")
multi_query_retriever = MultiQueryRetriever.from_llm(
retriever =retriever , llm=llm
)
query = "what is gradient descent?"
docs = multi_query_retriever.invoke(query)
print("\nRetriever Documents")
for doc in docs:
    print(doc.page_content)