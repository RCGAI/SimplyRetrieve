# SimplyRetrieve: A Private and Lightweight Retrieval-Centric Generative AI Tool
[![DOI](https://zenodo.org/badge/670095284.svg)](https://zenodo.org/badge/latestdoi/670095284)

## What is SimplyRetrieve?

![Tool Overall](docs/fig_overall.png)

This repository provides an implementation of *SimplyRetrieve*. SimplyRetrieve is an open-source tool with the goal of providing a fully localized, lightweight and user-friendly GUI and API for Retrieval-Centric Generation (RCG) architecture to the machine learning community. Some features are:
- GUI and API based Retrieval-Centric Generation platform
- Private Knowledge Base Constructor
- Retrieval Tuning Module with Prompts Engineering, Tool Configuration and Retrieval Analysis
- Fully localized, private and lightweight access to various sizes of Large Language Models (LLMs) within the retrieval-centric architecture
- Multi-user concurrent access to UI, utilizing Gradio's queue function

This tool is constructed based mainly on the awesome libraries of [Huggingface](https://huggingface.co/), [Gradio](https://gradio.app/), [PyTorch](https://pytorch.org/). The default LLM configured in this tool is the instruction-fine-tuned [Wizard-Vicuna-13B-Uncensored](https://huggingface.co/ehartford/Wizard-Vicuna-13B-Uncensored). The default embedding model for retriever is [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base). We found these models work well in this system, as well as many other various sizes of open-source LLMs and retrievers available in Huggingface. This tool can be run in other languages as well apart from English, by selecting appropriate LLMs and customizing prompt templates according to the target language.

A technical report about this tool will be made available soon at arXiv.

A short video about this tool is available at https://youtu.be/2N3NAi0N1TY

## Why this tool?
We aim to contribute to the development of safe, interpretable, and responsible LLMs by sharing our open-source tool for implementing RCG architectures. We hope this tool enables machine learning community to explore the use of LLMs in a more efficient way, while maintaining privacy and local implementation. By emphasizing the crucial role of the LLMs in context interpretation and entrusting knowledge memorization to the retriever, this approach has the potential to produce more efficient and interpretable generation, and reduce the scale of LLMs required for generative tasks. This tool can be run on a single Nvidia GPU, such as the A100, making it accessible to a wide range of users.

## What potential developments that can be explored using this tool?

- Examining the effectiveness of retrieval-centric generation in developing safer, more interpretable, and responsible AI systems
- Optimizing the efficiency of separating context interpretation and knowledge memorization within retrieval-centric generation architectures
- Improving prompt engineering techniques for Retrieval-Centric Generation
- Implementing Chat AI for edge computing and fully local implementations
- Creating personalized AI assistants tailored to individual users

## What are the prerequisites to run this tool?
- Git clone this repository.
- In GPU-based Linux machine, activate your favourite python venv, install the necessary packages
    - `pip install -r requirements.txt`
- If you would like to use your own data as a knowledge source, you can follow these steps. However, if you prefer to start with a simpler example, you can skip these steps and use the default simple sample knowledge source provided by the tool. Note that the sample knowledge source is intended only for demonstration purposes and should not be used for performance evaluations. To achieve accurate results, it is recommended to use your own knowledge source or the Wikipedia source for general usage.
    - Prepare knowledge source for retrieval: Put related documents(pdf etc.) into `chat/data/` directory and run the data preparation script (`cd chat/` then the following command) 
      ```
      CUDA_VISIBLE_DEVICES=0 python prepare.py --input data/ --output knowledge/ --config configs/default_chat1.json
      ```
    - Supported document formats are `pdf, txt, doc, docx, ppt, pptx, html, md, csv`, and can be easily expanded by editing configuration file

## How to run this tool?
After setting up the prerequisites above, set the current path to `chat` directory (`cd chat/`), execute the command below. Then grab a coffee! as it will just take a few minutes to load.

```
CUDA_VISIBLE_DEVICES=0 python chat.py --config configs/default_release.json
```
Then, access the web-based GUI from your favorite browser by navigating to `http://<LOCAL_SERVER_IP>:7860`. Replace `<LOCAL_SERVER_IP>` with the IP address of your GPU server. And this is it, you are ready to go! Below is a sample chat screenshot of the GUI. It provides a familiar streaming chatbot interface with a comprehensive retrieval-centric tuning panel. For API access, please refer to the sample scripts located in the `examples/` directory.

![Platform GUIsample](docs/gui_english.png)

Feel free to give us any feedback and comment. We welcome any discussion about this tool.

## Limitation
It is important to note that this tool does not provide a foolproof solution for ensuring a completely safe and responsible response from generative AI models, even within a retrieval-centric architecture. The development of safer, interpretable, and responsible AI systems remains an active area of research and ongoing effort.

Generated texts from this tool may exhibit variations, even when only slightly modifying prompts or queries, due to the next token prediction behavior of current-generation LLMs. This means users may need to carefully fine-tune both the prompts and queries to obtain optimal responses.

## Citation
If you find our work useful, please cite us as follow:
```
@software{Ng_SimplyRetrieve_A_Private_2023,
author = {Ng, Youyang and Miyashita, Daisuke and Hoshi, Yasuto and Morioka, Yasuhiro and Torii, Osamu and Kodama, Tomoya and Deguchi, Jun},
doi = {10.5281/zenodo.8213550},
month = jul,
title = {{SimplyRetrieve: A Private and Lightweight Retrieval-Centric Generative AI Tool}},
url = {https://github.com/RCGAI/SimplyRetrieve},
year = {2023}
}
```
Affiliation: Kioxia Corporation, Japan
