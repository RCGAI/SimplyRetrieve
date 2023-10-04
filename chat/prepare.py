# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import glob
import pandas as pd
import csv
import json
import importlib
import numpy as np
from typing import List
from tqdm import tqdm
from langchain.schema import Document
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--input', type=str, default='data/')
parser.add_argument('--output', type=str, default='knowledge/')
parser.add_argument('--split_encoder', type=str, default='gpt2')
parser.add_argument('--split_chunk_size', type=int, default=100)
parser.add_argument('--split_chunk_overlap', type=int, default=0)
parser.add_argument('--out_docs', type=str, default='local_knowledgebase')
parser.add_argument('--out_docsext', type=str, default='.tsv')

parser.add_argument('--do_embed', type=bool, default=True)
parser.add_argument('--embed_encoder', type=str, default='intfloat/multilingual-e5-base')
parser.add_argument('--out_embed', type=str, default='local_embed')
parser.add_argument('--out_embedext', type=str, default='.npy')

parser.add_argument('--index_method', type=str, default='hnsw')
parser.add_argument('--index_hnsw_m', type=int, default=64)
parser.add_argument('--out_index', type=str, default='local_index')
parser.add_argument('--out_indexext', type=str, default='.index')

parser.add_argument('--index_ivfpq_nlist', type=int, default=256)
parser.add_argument('--index_ivfpq_nsegment', type=int, default=16)
parser.add_argument('--index_ivfpq_nbit', type=int, default=8)
args, unknown = parser.parse_known_args()

os.makedirs(args.output, exist_ok=True)

def initialize_config(config):
    with open(config, "r", encoding="utf-8") as reader:
        text = reader.read()
    kwargs = json.loads(text)
    return kwargs

def initialize_loaders(kwargs):
    for k, v in kwargs.items():
        if k == 'loader_config':
            for kk, vv in v.items():
                if kk == 'ext_types':
                    for kkk, vvv in vv.items():
                        if isinstance(vvv, str) and vvv.startswith('langchain.'):
                            loader_module = importlib.import_module('.'.join(vvv.split('.')[:-1]))
                            loader = getattr(loader_module, vvv.split('.')[-1])
                            kwargs[k][kk][kkk] = loader

def docslist_load_local(directory):
    ext = "*.pdf"
    docslist = glob.glob(os.path.join(directory, ext), recursive=True)
    return docslist

def docslist_load_config(directory, loaders):
    docslist = []
    for ext, loader in loaders.items():
        docslist.extend(glob.glob(os.path.join(directory, "*"+ext), recursive=True))
    return docslist

def documents_load_local(doc_path):
    docs = []
    docs.extend(PyMuPDFLoader(doc_path).load())
    return docs

def documents_load_config(doc_path, loaders):
    docs = []
    docs.extend(loaders['.'+doc_path.split('.')[-1]](doc_path).load())
    return docs

def documents_load_file(doc_file, loaders):
    docs = []
    docs.extend(loaders['.'+doc_file.name.split('.')[-1]](doc_file.name).load())
    return docs

def documents_split(docs, encoder, chunk_size, chunk_overlap):
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splitter =  TokenTextSplitter(encoding_name=encoder, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_split = text_splitter.split_documents(docs)
    return docs_split

def documents_save(docs_split, path, filename, fileformat, index_current):
    global cnt_save
    savepath = os.path.join(path, filename+fileformat)
    if fileformat == ".tsv":
        docs_page_content = [o.page_content for o in docs_split]
        docs_page_content = pd.DataFrame(docs_page_content, columns=["page_content"])
        docs_page_content = docs_page_content.replace(r'-\n','', regex=True)
        docs_page_content = docs_page_content.replace(r'\n',' ', regex=True)
        docs_page_content = docs_page_content.replace(r'\t','  ', regex=True)
        docs_source = [o.metadata for o in docs_split]
        docs_source = pd.DataFrame(docs_source, columns=["source"])
        docs_out = pd.concat([docs_page_content, docs_source], axis=1)
        docs_out.index += 1
        if cnt_save == 0:
            writemode = 'w'
            cnt_save += 1
        else:
            docs_out.index += index_current
            writemode = 'a'
        docs_out.to_csv(savepath, index=True, header=False, sep='\t', mode=writemode)
    else:
        raise NotImplementedError("Not implemented at this time")
    return

def embedding_create(encoder, path, filename, fileformat):
    loadpath = os.path.join(path, filename+fileformat)
    docs_split = pd.read_csv(loadpath, sep='\t', header=None)
    print("total number of split passages for embedding:", len(docs_split))
    embeddings = HuggingFaceEmbeddings(model_name=encoder, encode_kwargs={'normalize_embeddings': True})
    embed_split = embeddings.embed_documents(docs_split[1])
    return embed_split

def embedding_save(embed_split, path, filename, fileformat):
    savepath = os.path.join(path, filename+fileformat)
    np.save(savepath, embed_split)
    return

def index_create(embed_split, method, hnsw_m, nlist, nsegment, nbit):
    if method == 'flat':
        index_split = faiss.IndexFlat(embed_split.shape[1], faiss.METRIC_INNER_PRODUCT)
    elif method == 'hnsw':
        index_split = faiss.IndexHNSWFlat(embed_split.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    elif method == 'indexivfpq_hnsw':
        coarse_split = faiss.IndexHNSWFlat(embed_split.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index_split = faiss.IndexIVFPQ(coarse_split, embed_split.shape[1], nlist, nsegment, nbit)
        index_split.train(embed_split)
    else:
        raise NotImplementedError("Not implemented at this time")
    index_split.add(embed_split)
    return index_split

def index_save(index_split, path, filename, fileformat):
    savepath = os.path.join(path, filename+fileformat)
    faiss.write_index(index_split, savepath)
    return

def insert_knowledge(config, k_dir, k_basename, k_disp, k_desc):
    config_new = config
    for item in config_new["retriever_config"]["retriever"]:
        if item["knowledgebase"] == os.path.join(k_dir, k_basename + ".tsv"):
            item["name"] = k_disp
            item["description"] = k_desc
            return config_new
    new_knowledge = {"name": k_disp,
                     "description": k_disp,
                     "knowledgebase": os.path.join(k_dir, k_basename + ".tsv"),
                     "index": os.path.join(k_dir, k_basename + ".index"),
                     "index_type": "hnsw"
                    }
    config_new["retriever_config"]["retriever"].append(new_knowledge)
    return config_new

def upload_knowledge(config, path_files, k_dir, k_basename, progress=gr.Progress()):
    global cnt_save
    if path_files == None:
        if os.path.exists(os.path.join(k_dir, k_basename+args.out_docsext)):
            return "Loaded existing knowledge"
        else:
            return "No knowledge to load"
    if os.path.exists(os.path.join(k_dir, k_basename+args.out_docsext)):
        cnt_save = 1
    else:
        cnt_save = 0
    progress(0.1, desc="Preparing")
    os.makedirs(k_dir, exist_ok=True)
    kwargs = config
    print("configs:", kwargs)
    initialize_loaders(kwargs)
    docslist = path_files
    print("total number of readable documents:", len(docslist))
    print("readable documents:", docslist)

    progress(0.3, desc="Loading, Splitting and Saving Documents")
    print("loading, splitting and saving documents...")
    cnt_passage = 0
    cnt_split = 0
    for item in tqdm(docslist):
        docs = documents_load_file(item, kwargs['loader_config']['ext_types'])
        docs_split = documents_split(docs, args.split_encoder, args.split_chunk_size, args.split_chunk_overlap)
        documents_save(docs_split, k_dir, k_basename, args.out_docsext, cnt_split)
        cnt_passage += len(docs)
        cnt_split += len(docs_split)
    print("total number of loaded passages:", cnt_passage)
    print("total number of split passages:", cnt_split)

    progress(0.5, desc="Creating and Saving Embedding")
    print("creating embedding")
    embed_split = embedding_create(kwargs['retriever_config']['encoder'], k_dir, k_basename, args.out_docsext)
    print("total number of embeddings:", len(embed_split))
    print("saving embedding")
    embedding_save(embed_split, k_dir, k_basename, args.out_embedext)

    progress(0.8, desc="Creating and Saving Index")
    print("creating index")
    embed_split = np.array(embed_split)
    index_split = index_create(embed_split, args.index_method, args.index_hnsw_m, args.index_ivfpq_nlist, args.index_ivfpq_nsegment, args.index_ivfpq_nbit)
    print("total number of indexes:", index_split.ntotal)
    print("saving index")
    index_save(index_split, k_dir, k_basename, args.out_indexext)

    print("documents preparation completed")

    return "New knowledge loaded"

def main():
    print(args)

    print("loading documents list...")
    if args.config == None:
        docslist = docslist_load_local(args.input)
    else:
        print("use config file")
        kwargs = initialize_config(args.config)
        print("configs:", kwargs)
        initialize_loaders(kwargs)
        docslist = docslist_load_config(args.input, kwargs['loader_config']['ext_types'])
    docslist.sort()
    print("total number of readable documents:", len(docslist))
    print("readable documents:", docslist)

    print("loading, splitting and saving documents...")
    cnt_passage = 0
    cnt_split = 0
    if args.config == None:
        for item in tqdm(docslist):
            docs = documents_load_local(item)
            docs_split = documents_split(docs, args.split_encoder, args.split_chunk_size, args.split_chunk_overlap)
            documents_save(docs_split, args.output, args.out_docs, args.out_docsext, cnt_split)
            cnt_passage += len(docs)
            cnt_split += len(docs_split)
    else:
        for item in tqdm(docslist):
            docs = documents_load_config(item, kwargs['loader_config']['ext_types'])
            docs_split = documents_split(docs, args.split_encoder, args.split_chunk_size, args.split_chunk_overlap)
            documents_save(docs_split, args.output, args.out_docs, args.out_docsext, cnt_split)
            cnt_passage += len(docs)
            cnt_split += len(docs_split)
    print("total number of loaded passages:", cnt_passage)
    print("total number of split passages:", cnt_split)

    if args.do_embed:
        print("creating embedding")
        embed_split = embedding_create(args.embed_encoder, args.output, args.out_docs, args.out_docsext)
        print("total number of embeddings:", len(embed_split))
        print("saving embedding")
        embedding_save(embed_split, args.output, args.out_embed, args.out_embedext)
        print("creating index")
        embed_split = np.array(embed_split)
        index_split = index_create(embed_split, args.index_method, args.index_hnsw_m, args.index_ivfpq_nlist, args.index_ivfpq_nsegment, args.index_ivfpq_nbit)
        print("total number of indexes:", index_split.ntotal)
        print("saving index")
        index_save(index_split, args.output, args.out_index, args.out_indexext)
        
    print("documents preparation completed")

if __name__ == "__main__":
    cnt_save = 0
    main()
