from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

docs =[Document(page_content="gardient docuemnt is an optimizatino algorithm used "),
       Document(page_content="gardient is minimize the loss "),
       Document(page_content= "gradient descent is organise and minimizes the loss"),
       Document(page_content= "Neural network use gradient decent for training."),
       Document(page_content = "support vector machine are supervised leaning algorithm")
]
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vectorstore = Chroma.from_documents(docs, embeddings)

similarity_retriever = vectorstore.as_retriever(
    search_type ="similarity",
    search_kwargs={"k":3})
print("\n==== similarity search=====")
similarity_docs = similarity_retriever.invoke("what is gradient desc")

for doc in similarity_docs:
    print(doc.page_content)
    
    
mmr_retriever = vectorstore.as_retriever(
    search_type ="mmr",
    search_kwargs={"k":3}
    )
print("\n====MMR search===")
mmr_docs = mmr_retriever.invoke("what is gradient descent ")
for doc in mmr_docs:
    print(doc.page_content)
