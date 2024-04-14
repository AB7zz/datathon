from getpass import getpass
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import csv_loader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Replicate
import os

REPLICATE_API_TOKEN = getpass()
print('working')
os.environ["REPLICATE_API_TOKEN"] = "r8_bnHCjky3IBgM9SeCIvQj2ebmBTHBD0R25bXjX"
llama2_13b = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
llm = Replicate(
    model=llama2_13b,
    model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
)

question = "I have data with columns: latitutde, longitude"
answer = llm(question)
print(answer)

# chat history not passed so Llama doesn't have the context and doesn't know this is more about the book
followup = "tell me more"
followup_answer = llm(followup)
print(followup_answer)

# DATA_PATH = 'data' #Your root data folder path
# DB_FAISS_PATH = 'vectorstore/db_faiss'