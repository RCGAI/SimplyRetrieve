# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import faiss
import sentence_transformers
from typing import Any, Dict, List, Optional

class CustomEmbeddings():
    def __init__(self, **kwargs: Any):
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {}
        self.__dict__.update(kwargs)
        self.model = sentence_transformers.SentenceTransformer(self.model_name, **self.model_kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.model.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        embedding = self.model.encode(text, **self.encode_kwargs)
        return embedding.tolist()

def embedding_create(encoder, query):
    embed_query = encoder.embed_query(query)
    return embed_query

def embedding_initialize(encoder):
    embedder = CustomEmbeddings(model_name=encoder, encode_kwargs={'normalize_embeddings': True})
    return embedder

def index_retrieve(index, embed_query, num_passage):
    d, i = index.search(embed_query, num_passage)
    return i

def knowledge_load_local(path):
    knowledge = pd.read_csv(path, sep='\t', header=None)
    return knowledge

def index_load_local(path, method):
    index = faiss.read_index(path)
    if "ivfpq" in method:
        print('current nprobe: {}'.format(index.nprobe))
        index.nprobe = 256
        print('changed to: {}'.format(index.nprobe))
        if "hnsw" in method:
            quantizer = faiss.downcast_index(index.quantizer)
            print('current efSearch: {}'.format(quantizer.hnsw.efSearch))
            faiss.downcast_index(index.quantizer).hnsw.efSearch = 128
            quantizer = faiss.downcast_index(index.quantizer)
            print('changed to: {}'.format(quantizer.hnsw.efSearch))
    elif "hnsw" in method:
        print('current efSearch: {}'.format(index.hnsw.efSearch))
        index.hnsw.efSearch = 128
        print('changed to: {}'.format(index.hnsw.efSearch))
    return index

def retriever_mokb(embed_mode, embed_query):
    similarity = np.dot(embed_mode, np.transpose(embed_query)).squeeze()
    print("MoKB similarity scores:", similarity, "Selected KB:", np.argmax(similarity))
    idx = np.argmax(similarity)
    return idx

def retriever_weighting(docs, weight):
    if weight == 100:
        return docs
    else:
        length = int(len(docs)*(weight/100))
        return docs[:length]

def initialize_retriever(kwargs):
    print("enable retriever and loading knowledge...")
    retriever_name = []
    retriever_desc = []
    knowledge = []
    index = []
    retriever_mode = ["Selectable KnowledgeBase", "Mixture-of-KnowledgeBase"]
    embed_mode = []
    for item in kwargs["retriever_config"]["retriever"]:
        retriever_name.append(item["name"])
        print("knowledge name:", item["name"])
        retriever_desc.append(item["description"])
        print("knowledge description:", item["description"])
        knowledge.append(knowledge_load_local(item["knowledgebase"]))
        print("total number of loaded passages:", len(knowledge[-1]))
        index.append(index_load_local(item["index"], item["index_type"]))
        print("total number of indexes:", index[-1].ntotal)
    encoder = embedding_initialize(kwargs["retriever_config"]["encoder"])
    for item in retriever_desc:
        embed_mode.append(embedding_create(encoder, item))
    return retriever_name, knowledge, index, encoder, retriever_mode, embed_mode
