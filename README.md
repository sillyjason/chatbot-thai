# RAG Chatbot with Couchbase

Building a enterprise-grade RAG application goes beyond merely vector search. Enterprises today generally suffer from these issues when trying to push their app to production or scale: 

This repo is a variance of [chatbot-cb-2](https://github.com/sillyjason/chatbot-cb-2), with slight modification on how embedding is generated: instead of calling OpenAI embedding API, we leveraged SentenceTransformer and "mrp/simcse-model-m-bert-thai-cased" embedding model where embedding inference is locally.
