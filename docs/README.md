## GUI Operation Manual

### Submit Query
- In the text box under Chat Tab, type in your query (question for Generative AI tool) and press Enter.

### Update Prompts
- In the text editor boxes under Prompt Tab, edit the content, then click the update button.

### Save Prompts
- Under Prompt Tab, edit the content, then click the save button.

### Update Configs
- In the text editor box under Config Tab, edit the content, then click the update button. Note that you can also directly modify the config file `configs/default_release.json` before running the tool. However, update configs through GUI is a convenient way in experimenting with the tool.
- Changing knowledge bases, LLMs etc. can be done through this Tab.

Selected items in configs:
- To use another LLM, modify the row of `"model_type": "ehartford/Wizard-Vicuna-13B-Uncensored",`. It is compatible with many models available on Hugging Face.
- To use another Dense Retriever, modify the row of `"encoder": "intfloat/multilingual-e5-base",`. It is compatible with many retrievers available on Hugging Face.
- To limit maximum length of generated response, modify the row of `"max_new_tokens": 512,`.
- To change the number of passages retrieved from knowledge base, modify the row of `"npassage": 5`.

### Save Configs
- Under Config Tab, edit the content, then click the save button
