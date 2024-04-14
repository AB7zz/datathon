from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
from langchain import PromptTemplate,  LLMChain
import os

# Set the Hugging Face API token
os.environ["HF_HOME"] = "C:/Users/abbas/Desktop/datathon/huggingface_hub"  # replace with your actual path

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    eos_token_id='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'
)
llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

template = """
              You are an intelligent chatbot that gives out useful information to humans.
              You return the responses in sentences with arrows at the start of each sentence
              {query}
           """

prompt = PromptTemplate(template=template, input_variables=["query"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run('What are the 3 causes of glacier meltdowns?'))