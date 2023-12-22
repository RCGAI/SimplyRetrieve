# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json
import gradio as gr
import numpy as np
import pandas as pd
from threading import Thread
from llms.initialize import initialize_llm
from prompts.initialize import initialize_prompts, save_prompts
from retrieval.retrieve import embedding_create, index_retrieve
from retrieval.retrieve import retriever_mokb, retriever_weighting, initialize_retriever
from prepare import upload_knowledge, insert_knowledge
from configs.config import read_config, save_config

# Command Line Arguments Setting
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/default_chat1.json')
parser.add_argument('--retriever', type=int, default=1)
parser.add_argument('--retchk', type=int, default=1)
parser.add_argument('--api', type=int, default=1)
parser.add_argument('--concurrencycount', type=int, default=1)
parser.add_argument('--fullfeature', type=int, default=1)
parser.add_argument('--logging', type=int, default=1)
parser.add_argument('--streaming', type=int, default=1)
args = parser.parse_args()

# Read Config File Function
kwargs = read_config(args.config)

# Initialize Prompts
kwargs = initialize_prompts(kwargs)

# Initialize LLM
main_agent_llm, streamer = initialize_llm(**kwargs['llm_config'])

# Initialize Retriever
retriever_chk = 0
retriever_name = ["None"]
retriever_mode = ["None"]
if args.retriever:
    retriever_name, knowledge, index, encoder, retriever_mode, embed_mode = initialize_retriever(kwargs)
    retriever_chk = args.retchk

# Initialize Data Logging
def initialize_logs():
    df = pd.DataFrame(columns=['Sim_QKSL', 'Sim_RKSL', 'Sim_QKTL', 'Sim_RKTL', 'Query', 'Response', 'Prompt'])
    return df
df = initialize_logs()

# Update Prompts Function
def update_prompts(*prompts):
    global kwargs
    kwargs['prompt_config']['prompt_prefix'] = prompts[0]
    kwargs['prompt_config']['prompt_suffix'] = prompts[1]
    if args.retriever:
        kwargs['prompt_config']['prompt_retrieveprefix'] = prompts[2]
        kwargs['prompt_config']['prompt_retrievesuffix'] = prompts[3]
    kwargs['prompt_config']['prompt_aiprefix'] = prompts[4]
    kwargs = initialize_prompts(kwargs)
    print("Prompts update completed")

# Update Configs Function
def update_config(progress=gr.Progress(), *config):
    global kwargs, main_agent_llm, knowledge, index, encoder
    progress(0.1, desc="Loading Config")
    kwargs_new = json.loads(config[0])
    kwargs = kwargs_new
    progress(0.2, desc="Initializing Prompts")
    kwargs = initialize_prompts(kwargs)
    update_prompts(*config[1:])
    progress(0.3, desc="Initializing LLM")
    main_agent_llm = None
    main_agent_llm, streamer = initialize_llm(**kwargs['llm_config'])
    progress(0.8, desc="Initializing Knowledge Base")
    if args.retriever:
        retriever_name, knowledge, index, encoder, retriever_mode, embed_mode = initialize_retriever(kwargs)
    retriever_name_tuple = []
    for item in retriever_name:
        retriever_name_tuple.append((item, item))
    print("Update Config Completed")
    return "Update Complete", gr.Dropdown(choices=retriever_name_tuple)

# Update KnowledgeBase
def update_knowledge(config, path_files, k_dir, k_basename, k_disp, k_desc, progress=gr.Progress()):
    global kwargs, retriever_name, knowledge, index, encoder, retriever_mode, embed_mode, drop_retriever
    config_loaded = json.loads(config)
    status = upload_knowledge(kwargs, path_files, k_dir, k_basename, progress)
    if status != "No knowledge to load":
        config_loaded = insert_knowledge(config_loaded, k_dir, k_basename, k_disp, k_desc)
        kwargs["retriever_config"]["retriever"] = config_loaded["retriever_config"]["retriever"]
        retriever_name, knowledge, index, encoder, retriever_mode, embed_mode = initialize_retriever(kwargs)
    retriever_name_tuple = []
    for item in retriever_name:
        retriever_name_tuple.append((item, item))
    return status, gr.Dropdown(choices=retriever_name_tuple), json.dumps(config_loaded, indent=4)

# Load Logs
def load_logs():
    return df

# Save Logs
def save_logs(path):
    df.to_csv(path, index=False)
    print("Save Logs Completed")
    return

def main():
    # SimplyRetrieve App
    with gr.Blocks(title="SimplyRetrieve Chat AI") as app:
        # Chat UI Tab
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(elem_id="chatbot")
            with gr.Row():
                msg = gr.Textbox(show_label=False, placeholder="Enter text and press enter", container=False)
            api_msg = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=False, container=False)
            api_msg_llm = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=False, container=False)
            with gr.Row():
                chkbox_retriever = gr.Checkbox(label="Use KnowledgeBase", value=retriever_chk, visible=args.retriever)
                drop_retmode = gr.Dropdown(retriever_mode, value=retriever_mode[0], type="index", multiselect=False, visible=args.retriever, label="KnowledgeBase Mode")
                drop_retriever = gr.Dropdown(retriever_name, value=retriever_name[0], type="value", multiselect=False, visible=args.retriever, label="KnowledgeBase")
                with gr.Column():
                    chkbox_retweight = gr.Checkbox(label="Prompt-Weighting", value=0, visible=args.retriever)
                    slider_retweight = gr.Slider(label="KnowledgeBase Weightage", minimum=0, maximum=100, value=100, step=1)

            def user(user_message, history):
                if history is None:
                    history = ""
                return "", history + [[user_message, None]]

            # GUI Query Processing
            def bot(history, *retriever):
                # Arguments Processing
                global kwargs, df
                prompt = history[-1][0]
                flag_retrieve = retriever[0]
                mode = retriever[1]
                if mode == None:
                    mode = 0
                #idx = retriever[2]
                idx = retriever_name.index(retriever[2])
                if idx == None:
                    idx = 0
                flag_weight = retriever[3]
                weight = retriever[4]
                flag_logging = retriever[5]

                # Query & KnowledgeBase Processing
                if flag_retrieve:
                    embed_query = embedding_create(encoder, prompt)
                    if mode == 1: # Mixture-of-KnowledgeBase
                        idx = retriever_mokb(embed_mode, embed_query)
                    index_result = index_retrieve(index[idx], np.array([embed_query]), kwargs["retriever_config"]["npassage"])
                    docs = [knowledge[idx][1][ii] for ii in index_result[0]]
                    docs_joined = ' '.join(docs)
                    docs = docs_joined
                    if flag_weight:
                        docs = retriever_weighting(docs, weight)
                    docs = kwargs['prompt_config']['prompt_retrieveprefix'] + docs + kwargs['prompt_config']['prompt_retrievesuffix']
                    textin = kwargs['prompt_config']['prompt_aiprefix'] + docs + kwargs['prompt_config']['prompt_prefix'] + prompt + kwargs['prompt_config']['prompt_suffix']
                else:
                    textin = kwargs['prompt_config']['prompt_aiprefix'] + kwargs['prompt_config']['prompt_prefix'] + prompt + kwargs['prompt_config']['prompt_suffix']

                # LLM Processing
                print("User: " + '\033[32m' + textin + '\033[0m')
                if args.streaming:
                    generation_kwargs = dict(prompts=textin, streamer=streamer)
                    thread = Thread(target=main_agent_llm, kwargs=generation_kwargs)
                    thread.start()
                    res = ""
                    for new_text in streamer:
                        res += new_text
                        history[-1][1] = res
                        yield history, "", "", "", "", "", "", ""
                else:
                    res = main_agent_llm(textin)
                    history[-1][1] = res
                print("AI: " + '\033[35m' + res + '\033[0m')

                # Response Post Processing
                if flag_retrieve:
                    embed_joined = embedding_create(encoder, docs_joined)
                    embed_res = embedding_create(encoder, res)
                    res_slscores = np.dot(embed_joined, np.transpose(embed_res)).squeeze()
                    query_slscores = np.dot(embed_joined, np.transpose(embed_query)).squeeze()

                    encoder.encode_kwargs["output_value"] = "token_embeddings"
                    tokens_joined = np.array(embedding_create(encoder, docs_joined))
                    tokens_joined = tokens_joined / np.linalg.norm(tokens_joined, axis=-1, keepdims=True)
                    tokens_query = np.array(embedding_create(encoder, prompt))
                    tokens_query = tokens_query / np.linalg.norm(tokens_query, axis=-1, keepdims=True)
                    tokens_res = np.array(embedding_create(encoder, res))
                    tokens_res = tokens_res / np.linalg.norm(tokens_res, axis=-1, keepdims=True)
                    encoder.encode_kwargs["output_value"] = "sentence_embedding"
                    res_tlscores = np.average(np.dot(tokens_joined, np.transpose(tokens_res)).squeeze())
                    query_tlscores = np.average(np.dot(tokens_joined, np.transpose(tokens_query)).squeeze())
                else:
                    res_slscores = 0
                    query_slscores = 0
                    res_tlscores = 0
                    query_tlscores = 0

                if flag_logging:
                    new_data = {'Sim_QKSL':query_slscores, 'Sim_QKTL':query_tlscores, 'Sim_RKSL':res_slscores, 
                            'Sim_RKTL':res_tlscores, 'Query':prompt, 'Response':res, 'Prompt':textin}
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                    if len(df.index) > 1000:
                        df = df.drop([0])

                if args.streaming:
                    yield history, prompt, textin, res, res_slscores, query_slscores, res_tlscores, query_tlscores
                    return
                else:
                    yield history, prompt, textin, res, res_slscores, query_slscores, res_tlscores, query_tlscores
                    return

            # API for Query
            def api_user_bot(user_message, *retriever):

                # Arguments Processing
                global kwargs, df
                prompt = user_message
                flag_retrieve = retriever[0]
                mode = retriever[1]
                if mode == None:
                    mode = 0
                #idx = retriever[2]
                idx = retriever_name.index(retriever[2])
                if idx == None:
                    idx = 0
                flag_weight = retriever[3]
                weight = retriever[4]
                res_scores = 0
                flag_logging = retriever[5]

                # Query & KnowledgeBase Processing
                if flag_retrieve:
                    embed_query = embedding_create(encoder, prompt)
                    if mode == 1: # Mixture-of-KnowledgeBase
                        idx = retriever_mokb(embed_mode, embed_query)
                    index_result = index_retrieve(index[idx], np.array([embed_query]), kwargs["retriever_config"]["npassage"])
                    docs = [knowledge[idx][1][ii] for ii in index_result[0]]
                    docs_joined = ' '.join(docs)
                    docs = docs_joined
                    if flag_weight:
                        docs = retriever_weighting(docs, weight)
                    docs = kwargs['prompt_config']['prompt_retrieveprefix'] + docs + kwargs['prompt_config']['prompt_retrievesuffix']
                    textin = kwargs['prompt_config']['prompt_aiprefix'] + docs + kwargs['prompt_config']['prompt_prefix'] + prompt + kwargs['prompt_config']['prompt_suffix']
                else:
                    textin = kwargs['prompt_config']['prompt_aiprefix'] + kwargs['prompt_config']['prompt_prefix'] + prompt + kwargs['prompt_config']['prompt_suffix']

                # LLM Processing
                print("User: " + '\033[32m' + textin + '\033[0m')
                if args.streaming:
                    generation_kwargs = dict(prompts=textin, streamer=streamer)
                    thread = Thread(target=main_agent_llm, kwargs=generation_kwargs)
                    thread.start()
                    res = ""
                    for new_text in streamer:
                        res += new_text
                else:
                    res = main_agent_llm(textin)
                print("AI: " + '\033[35m' + res + '\033[0m')

                # Response Post Processing
                if flag_retrieve:
                    embed_joined = embedding_create(encoder, docs_joined)
                    embed_res = embedding_create(encoder, res)
                    res_slscores = np.dot(embed_joined, np.transpose(embed_res)).squeeze()
                    query_slscores = np.dot(embed_joined, np.transpose(embed_query)).squeeze()

                    encoder.encode_kwargs["output_value"] = "token_embeddings"
                    tokens_joined = np.array(embedding_create(encoder, docs_joined))
                    tokens_joined = tokens_joined / np.linalg.norm(tokens_joined, axis=-1, keepdims=True)
                    tokens_query = np.array(embedding_create(encoder, prompt))
                    tokens_query = tokens_query / np.linalg.norm(tokens_query, axis=-1, keepdims=True)
                    tokens_res = np.array(embedding_create(encoder, res))
                    tokens_res = tokens_res / np.linalg.norm(tokens_res, axis=-1, keepdims=True)
                    encoder.encode_kwargs["output_value"] = "sentence_embedding"
                    res_tlscores = np.average(np.dot(tokens_joined, np.transpose(tokens_res)).squeeze())
                    query_tlscores = np.average(np.dot(tokens_joined, np.transpose(tokens_query)).squeeze())
                else:
                    res_slscores = 0
                    query_slscores = 0
                    res_tlscores = 0
                    query_tlscores = 0

                if flag_logging:
                    new_data = {'Sim_QKSL':query_slscores, 'Sim_QKTL':query_tlscores, 'Sim_RKSL':res_slscores,
                            'Sim_RKTL':res_tlscores, 'Query':prompt, 'Response':res, 'Prompt':textin}
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                    if len(df.index) > 1000:
                        df = df.drop([0])

                # return prompt, textin, res, res_slscores, query_slscores, res_tlscores, query_tlscores
                return res

            # API for Query with direct LLM access without prompts
            def api_user_llm(user_message):
                global kwargs
                textin = user_message
                print("User: " + '\033[32m' + textin + '\033[0m')
                if args.streaming:
                    generation_kwargs = dict(prompts=textin, streamer=streamer)
                    thread = Thread(target=main_agent_llm, kwargs=generation_kwargs)
                    thread.start()
                    res = ""
                    for new_text in streamer:
                        res += new_text
                else:
                    res = main_agent_llm(textin)
                print("AI: " + '\033[35m' + res + '\033[0m')
                return res

        # Prompt-Engineering Tab
        with gr.Tab("Prompt"):
            global kwargs
            if args.retriever:
                gr.Markdown("""Prompt = AI Prefix + Retriever Prefix + **Retrieved KnowledgeBase** +
                        Retriever Suffix + Model Prefix + **Query** + Model Suffix""")
            else:
                gr.Markdown("Prompt = AI Prefix + Model Prefix + **Query** + Model Suffix")
            with gr.Row():
                prompt_prefix = gr.TextArea(label='Model Prefix', value=kwargs['prompt_config']['prompt_prefix'], lines=3)
                prompt_suffix = gr.TextArea(label='Model Suffix', value=kwargs['prompt_config']['prompt_suffix'], lines=3)
                prompts = [prompt_prefix, prompt_suffix]
            with gr.Row():
                if args.retriever:
                    prompt_retrieveprefix = gr.TextArea(label='Retriever Prefix',
                            value=kwargs['prompt_config']['prompt_retrieveprefix'], lines=3)
                    prompt_retrievesuffix = gr.TextArea(label='Retriever Suffix',
                            value=kwargs['prompt_config']['prompt_retrievesuffix'], lines=3)
                    prompts.extend([prompt_retrieveprefix, prompt_retrievesuffix])
            with gr.Row():
                prompt_aiprefix = gr.TextArea(label='AI Prefix',
                    value=kwargs['prompt_config']['prompt_aiprefix'], lines=3)
                prompts.extend([prompt_aiprefix])
            with gr.Row():
                update_prompts_btn = gr.Button("Update Prompts", visible=args.fullfeature)
                with gr.Column():
                    gr.Markdown("*Prompts will be saved to subdirectory of prompts in separate files*", visible=args.fullfeature)
                    save_prompts_btn = gr.Button("Save Prompts", visible=args.fullfeature)

        # Configuration Tab
        with gr.Tab("Config"):
            with gr.Row():
                config_txt = gr.TextArea(label='Config File', value=json.dumps(read_config(args.config), indent=4))
            with gr.Row():
                update_config_btn = gr.Button("Update Config", visible=args.fullfeature)
                save_config_btn = gr.Button("Save Config", visible=args.fullfeature)
                name, ext = os.path.splitext(args.config)
                save_path_default = name + "_new" + ext
                save_path = gr.Textbox(label='Save Path', value=save_path_default, visible=args.fullfeature)
            progress_c = gr.Textbox(label='Progress', value="Ready", visible=args.fullfeature)

        # Analysis and Data-Logging Tab
        with gr.Tab("Analysis"):
            with gr.Row():
                chkbox_logging = gr.Checkbox(label="Data Logging", value=args.logging, visible=args.fullfeature)
            with gr.Row():
                with gr.Column():
                    anls_qkslscore = gr.Textbox(label='Query & KnowledgeBase Sentence Level Similarity Score')
                    anls_rkslscore = gr.Textbox(label='Response & KnowledgeBase Sentence Level Similarity Score')
                with gr.Column():
                    anls_qktlscore = gr.Textbox(label='Query & KnowledgeBase Tokens Level Similarity Score')
                    anls_rktlscore = gr.Textbox(label='Response & KnowledgeBase Tokens Level Similarity Score')
            with gr.Row():
                with gr.Column():
                    anls_query = gr.Textbox(label='Query')
                    anls_res = gr.Textbox(label='Response')
                anls_prompt = gr.TextArea(label='Prompt')
            with gr.Row():
                load_log_btn = gr.Button("Load Logs", visible=args.fullfeature)
                save_log_btn = gr.Button("Save Logs", visible=args.fullfeature)
                save_log_path = gr.Textbox(label='Save Log Path', value="analysis/logs.csv", visible=args.fullfeature)
            datahistory = gr.Dataframe(
                        label="Query and Response History",
                        headers=["Query", "Response"],
                        row_count = (5, "dynamic"),
                        col_count=(2,"dynamic"),
                        interactive=False, type="array", wrap=False)

        # KnowledgeBase Creation Tab
        with gr.Tab("Knowledge"):
            with gr.Row():
                files_knowledge = gr.File(file_count="multiple")
            gr.Markdown("If *KnowledgeBase Filename* existed, Knowledge will be inserted in Append Mode.")
            with gr.Row():
                dir_knowledgebase = gr.Textbox(label='KnowledgeBase Directory', value="knowledge/", visible=args.fullfeature)
                name_knowledgebase = gr.Textbox(label='KnowledgeBase Filename', value="knowledge_new", visible=args.fullfeature)
                disp_knowledgebase = gr.Textbox(label='KnowledgeBase Display Name', value="knowledge New", visible=args.fullfeature)
                desc_knowledgebase = gr.Textbox(label='KnowledgeBase Description', value="My personal knowledge", visible=args.fullfeature)
            btn_k_upload = gr.Button("Create Knowledge", visible=args.fullfeature)
            progress_k = gr.Textbox(label='Progress', value="Ready", visible=args.fullfeature)

        # Events of Chat AI
        retriever = [chkbox_retriever, drop_retmode, drop_retriever, chkbox_retweight, slider_retweight, chkbox_logging]
        analysis = [anls_query, anls_prompt, anls_res, anls_rkslscore, anls_qkslscore, anls_rktlscore, anls_qktlscore]
        msg.submit(user, [msg, chatbot], [msg, chatbot], concurrency_limit=args.concurrencycount).then(
                    bot, [chatbot, *retriever], [chatbot, *analysis], concurrency_limit=args.concurrencycount)
        api_msg.submit(api_user_bot, [api_msg, *retriever], api_msg, api_name="message-query")
        api_msg_llm.submit(api_user_llm, api_msg_llm, api_msg_llm, api_name="message-direct-llm")

        # Events of Prompt-Engineering
        update_prompts_btn.click(fn=update_prompts, inputs=prompts, api_name="update-prompts")
        save_prompts_btn.click(fn=save_prompts, inputs=prompts, api_name="save-prompts")

        # Events of Configurations
        config_current = [config_txt]
        config_current.extend(prompts)
        update_config_btn.click(fn=update_config, inputs=config_current, outputs=[progress_c, drop_retriever], api_name="update-config")
        save_config_btn.click(fn=save_config, inputs=[config_txt, save_path], outputs=progress_c, api_name="save-config")

        # Events of Analysis and Data Logging
        load_log_btn.click(fn=load_logs, inputs=None, outputs=datahistory, api_name="load-logs")
        save_log_btn.click(fn=save_logs, inputs=save_log_path, api_name="save-logs")

        # Events of KnowledgeBase Creation
        btn_k_upload.click(fn=update_knowledge, inputs=[config_txt, files_knowledge, dir_knowledgebase,
            name_knowledgebase, disp_knowledgebase, desc_knowledgebase],
            outputs=[progress_k, drop_retriever, config_txt], api_name="upload-knowledge")

    # App Main Settings
    app.queue(max_size=100, api_open=args.api)
    app.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
