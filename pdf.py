from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load PDF
data = PyPDFLoader("lbdl.pdf")
docs = data.load()
splitter = TokenTextSplitter(chunk_size = 100, chunk_overlap=10)
chunks = splitter.split_documents(docs)
print(chunks[10].page_content)

# Define Template
#template = ChatPromptTemplate.from_messages([
#    ("system", "you are a AI that summarizes text"),
#    ("human", "{data}")
#])

# Initialize Model with valid name
#model = ChatMistralAI(model="mistral-small-latest")

# Format and invoke model
#prompt = template.invoke({"data": docs[0].page_content})
#result = model.invoke(prompt)

#print(result.content)