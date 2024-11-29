from pymilvus.model.hybrid import BGEM3EmbeddingFunction  
import base64
from langchain.embeddings.base import Embeddings  

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_milvus.vectorstores import Milvus  


def encode_image(image_path):  
    with open(image_path, "rb") as image_file:  
        return base64.b64encode(image_file.read()).decode("utf-8")  


class BGEMilvusEmbeddings(Embeddings):  
    def __init__(self):  
        self.model = BGEM3EmbeddingFunction(  
                    model_name='BAAI/bge-m3',
                    device='cpu', 
                    use_fp16=False 
                )  
  
    def embed_documents(self, texts):  
  
        embeddings = self.model.encode_documents(texts)  
  
        return [i.tolist() for i in embeddings["dense"]]  
  
    def embed_query(self, text):  
  
        embedding = self.model.encode_queries([text])  
  
        return embedding["dense"][0].tolist()  
    
    
def insert_image_if_needed(message, retriever):
    if message["content"][0]["type"] != "image":
        query = message["content"][0]["text"]
        image = retriever.invoke(query, limit=1)[0].metadata["source"]
        message["content"].insert(0, {
            "type": "image",
            "image": f"data:image;base64,{image}",
        })
        
def process_messages(messages):
    URI = "/home/zhuyao/Sunpeng/llava_qwen/RAG/milvus_emo.db"  

    embedding_model = BGEMilvusEmbeddings()  
    vector_store = Milvus(
        embedding_function=embedding_model,
        collection_name="multimodal_rag_demo",
        connection_args={"uri": URI},
    )

    retriever = vector_store.as_retriever()  
    if isinstance(messages[0], list):
        for message in messages:
            if message[0]["role"] == "system":
                insert_image_if_needed(message[1], retriever)
            else:
                insert_image_if_needed(message[0], retriever)
    else:
        if messages[0]["role"] == "system":
            insert_image_if_needed(messages[1], retriever)
        else:
            insert_image_if_needed(messages[0], retriever)
    return messages

# if isinstance(messages[0], list):
#     for message in messages:
#         if message[0]["role"] == "system":
#             insert_image_if_needed(message[1], retriever)
#         else:
#             insert_image_if_needed(message[0], retriever)
# else:
#     if messages[0]["role"] == "system":
#         insert_image_if_needed(messages[1], retriever)
#     else:
#         insert_image_if_needed(messages[0], retriever)
# print(messages)