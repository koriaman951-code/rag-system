from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vectorstore = Chroma(
    persist_directory ="Chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_type = "mmr",
    search_kwargs={"k":4,"fetch_k":10,
    "lambda_mult":0.5})

llm = ChatMistralAI(model = "mistral-small-2506")
#prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """you are helpful ai asistant.
 Use onlythe provided context to answer the question.
 if the answer isi not present in the context, 
 say:"i could not find th answer in the document."
 
 """),
        ("human",
         """Context:
         {context}
question:
{question}
"""
         )
        ]
    )
print("rag system created")
print("press 0 to exit")
while True:
    query = input("you:")
    if query =="0":
        break
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.invoke({
        "context":context,
         "question":query                        
         })
    response = llm.invoke(final_prompt)
    print(f"AI:{response.content}")