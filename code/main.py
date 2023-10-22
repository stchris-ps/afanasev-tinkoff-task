from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import Optional
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

from translatepy import Translator

class PlaceOrder(BaseModel):
    skus: List[str] = Field(description="products, that user asked for")
    address: str = Field(description="delivery address")
    deliverytime : datetime = Field(description="delivery datetime")

parser = PydanticOutputParser(pydantic_object=PlaceOrder)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
      model_path="llama-2-7b-chat.Q4_K_M.gguf",
      temperature=0.25,
      max_tokens=2000,
      top_p=1,
      callback_manager=callback_manager,
      verbose=True, # Verbose is required to pass to the callback manager
      n_ctx=1200
)
app = FastAPI()
redis_backed_dict = {}

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a robot operator of Tinkoff Bank. You must answer customer questions only on Russian language. After the answer one question wait for the next question from the customer.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ],
#    partial_variables={
#        "menu": 'get_from_backend()',
#        "format_instructions": parser.get_format_instructions()
#        },
    output_language="Russian"
)

translator = Translator()

loader = DirectoryLoader('datas/', glob="*")
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 500,
    chunk_overlap = 50,
    length_function = len,
    is_separator_regex = False,
)
loader = DirectoryLoader('datas/', glob="*")
all_text = loader.load()

chunks = text_splitter.split_documents(all_text)#[0].page_content+uisd[1].page_content
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(chunks, embeddings_model)#

print('OK1')
print('\n')
@app.post("/message")
def message(user_id: str, message: str):
    memory = redis_backed_dict.get(
        user_id,
        ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
    question_en = translator.translate(message, "English").result
    docs = db.similarity_search(question_en)
    context = 'Для ответа на вопрос используй эту информацию:'
    for frag in docs[:5]:
        context += frag.page_content
    
    ai_message = conversation.run({"question": question_en+context})
    return {"message": ai_message["text"]}

print('OK2')
print('\n')