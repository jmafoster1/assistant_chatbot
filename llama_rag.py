#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:31:36 2025

@author: lesya
"""

from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

model_id = 'meta-llama/Meta-Llama-3-8B'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type='nf4',
    # bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=bfloat16
)

hf_access_token = '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'
from huggingface_hub import login
login(token=hf_access_token)


time_1 = time()
model_config = transformers.AutoConfig.from_pretrained(
    model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    # quantization_config=bnb_config,
    config=model_config,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
time_2 = time()
print(f"Prepare model, tokenizer: {round(time_2-time_1, 3)} sec.")




time_1 = time()
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",)
time_2 = time()
print(f"Prepare pipeline: {round(time_2-time_1, 3)} sec.")


print("loading the documents")
loader = TextLoader("/docs/data.pdf",
                    encoding="utf8")
documents = loader.load()




text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
all_splits = text_splitter.split_documents(documents)


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "auto"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


print("creating a database")
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")





def test_model(tokenizer, pipeline, prompt_to_test):
    """
    Perform a query
    print the result
    Args:
        tokenizer: the tokenizer
        pipeline: the pipeline
        prompt_to_test: the prompt
    Returns
        None
    """
    # adapted from https://huggingface.co/blog/llama2#using-transformers
    time_1 = time()
    sequences = pipeline(
        prompt_to_test,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,)
    time_2 = time()
    print(f"Test inference: {round(time_2-time_1, 3)} sec.")
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        
print("testing the model")        
        
# test_model(tokenizer,
#            query_pipeline,
#            "Please explain which metrics were used to evaluate the model performance.")

llm = HuggingFacePipeline(pipeline=query_pipeline)
# checking again that everything is working fine
llm(prompt="You are a conversational agent that helps the user to navigare through the output of the synthetic image analysis tool in the assistant plugin: https://weverify.eu/verification-plugin/. Please answer the following questions based on the documentation regarding the model information and the interface: ")





retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)



def test_rag(qa, query):
    print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    print("\nResult: ", result)
    
    
    
query = "What do gradients mean?"
test_rag(qa, query)
query = "That is the model performance?"
test_rag(qa, query)


docs = vectordb.similarity_search(query)
print(f"Query: {query}")
print(f"Retrieved documents: {len(docs)}")
for doc in docs:
    doc_details = doc.to_json()['kwargs']
    print("Source: ", doc_details['metadata']['source'])
    print("Text: ", doc_details['page_content'], "\n")
    
    

