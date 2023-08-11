## GUI Operation Manual

### Submit Query
- In the text box under Chat Tab, type in your query (question for Generative AI tool) and press Enter.

### Update Prompts
- In the text editor boxes under Prompt Tab, edit the content, then click the update button.

### Update Configs
- In the text editor box under Config Tab, edit the content, then click the update button.
- Changing knowledge bases, LLMs etc. can be done through this Tab.
- To use another LLM, modify the row of `"model_type": "ehartford/Wizard-Vicuna-13B-Uncensored",`. It is compatible with many available models on Hugging Face.
- To use another Dense Retriever, modify the row of `"encoder": "intfloat/multilingual-e5-base",`. It is compatible with many available retrievers on Hugging Face.
- To limit maximum length of generated response, modify the row of `"max_new_tokens": 512,`.
