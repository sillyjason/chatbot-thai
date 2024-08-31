from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from flask import jsonify


chat_openai = ChatOpenAI(model="gpt-4o", temperature=0.01)    

client_openai = OpenAI()

prompt_openai = ChatPromptTemplate.from_template("""Answer the following question incorporating the following context:
<context>
{context}
</context>

The answer should be precise and professional, and no longer than 5 sentences. Also answer the question in the language with which the question was asked. 

Question: {input}""")


embedding_model = SentenceTransformer('mrp/simcse-model-m-bert-thai-cased')

query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
        ),
    ]
)

def create_openai_embeddings(input_message):
    # return client_openai.embeddings.create(input = [input_message], model="text-embedding-ada-002").data[0].embedding
    # this function is being modified to use SentenceTransformer to generate embedding for Thai language
    
    # Generate the embedding
    embedding = embedding_model.encode(input_message)
    
    # Convert ndarray to list
    openai_embedding_list = embedding.tolist()
    
    # Return the list as JSON
    return openai_embedding_list



def generate_query_transform_prompt(messages):
    query_transformation_chain = query_transform_prompt | chat_openai
    
    print("generating transformed query...")
    return query_transformation_chain.invoke({"messages": messages}).content 
    

def generate_document_chain():     
    return create_stuff_documents_chain(chat_openai, prompt_openai)