from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from sentence_transformers import SentenceTransformer


chat_openai = ChatOpenAI(model="gpt-4o", temperature=0.01)    

client_openai = OpenAI()

prompt_openai = ChatPromptTemplate.from_template("""Answer the following question incorporating the following context:
<context>
{context}
</context>

The answer should be precise and professional, and no longer than 5 sentences. 

Question: {input}""")


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
    # Load the SimCSE model for Thai
    model = SentenceTransformer('mrp/simcse-model-m-bert-thai-cased')

    # Define the sentence
    sentence = "I love Canada"

    # Generate the embedding
    return model.encode(sentence)



def generate_query_transform_prompt(messages):
    query_transformation_chain = query_transform_prompt | chat_openai
    
    print("generating transformed query...")
    return query_transformation_chain.invoke({"messages": messages}).content 
    

def generate_document_chain():     
    return create_stuff_documents_chain(chat_openai, prompt_openai)