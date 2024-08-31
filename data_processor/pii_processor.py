from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json 


model = ChatOpenAI(model="gpt-4o")

parser = StrOutputParser()


prompt_template = PromptTemplate.from_template(
   """The input_text below is a json document. Scan the document, identify 
   all occurences of any sensitive information such as phone numbers, personal identifiable ids, addresss, etc.
   Then, mask the sensitive information by replacing it with a placeholder.
   For example, if the information is a personal id, replace last four digits with 'xxxx'. 
   if the information is a phone number, replace it with 'xxx-xxx-xxxx'.
   if the information is an email, replace it with xxxx@xxxx.xxxx'
   
   input_text: {input_text}
   Your output should only contain the transformed document, formatted into a JSON object, which begins and ends with curly brackets. 
   Delete all leading and trailing ``` characters, and "json" wordings outside the JSON object. 
   """
)

chain = prompt_template | model | parser

def mask_sensitive_data(request_data):    
    result = chain.invoke({"input_text": request_data})
    #transform result into a dictionary
    result_dict = json.loads(result)
    return result_dict