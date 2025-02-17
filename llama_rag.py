#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:31:36 2025

@author: lesya
"""

from torch import bfloat16
import torch
import transformers
from transformers import AutoTokenizer

from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from huggingface_hub import login

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
HF_ACCESS_TOKEN = "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"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class LLAMA_RAG:
    def __init__(self):

        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=bfloat16,
        )

        login(token=HF_ACCESS_TOKEN)

        model_config = transformers.AutoConfig.from_pretrained(MODEL_ID)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=bnb_config,
            config=model_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        query_pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print("loading the documents")
        loader = TextLoader("docs/data.txt", encoding="utf8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
        all_splits = text_splitter.split_documents(documents)

        model_kwargs = {"device": "cuda"}

        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME, model_kwargs=model_kwargs
        )

        print("creating a database")
        self.vectordb = Chroma.from_documents(
            documents=all_splits, embedding=embeddings, persist_directory="chroma_db"
        )

        print("testing the model")

        llm = HuggingFacePipeline(pipeline=query_pipeline)
        llm(
            prompt="You are a conversational agent that helps the user to navigare through the output of the synthetic image analysis tool in the assistant plugin: https://weverify.eu/verification-plugin/. Please answer the following questions based on the documentation regarding the model information and the interface: "
        )

        retriever = self.vectordb.as_retriever()

        self.qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, verbose=True
        )

    def call_rag(self, query):
        result = self.qa.run(query)
        docs = [
            doc.to_json()["kwargs"] for doc in self.vectordb.similarity_search(query)
        ]
        return result, docs


if __name__ == "__main__":
    llama_rag = LLAMA_RAG()
    query = "What do gradients mean?"
    print("=" * 40, query, "=" * 40)
    result, docs = llama_rag.call_rag(query)
    print(result, docs)
    query = "That is the model performance?"
    print("=" * 40, query, "=" * 40)
    result, docs = llama_rag.call_rag(query)
    print(result, docs)
