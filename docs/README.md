# GUI Operation Manual of SimplyRetrieve
This manual provides simple instructions on operating the GUI of this tool. The manual will be updated accordingly.

## Chat Tab

#### Ask Question to Get a Response
- In the text box, type in your query (question for Generative AI tool) and press `Enter`.

#### Enable/Disable Knowledge Base
- Check or uncheck the `Use Knowledgebase` checkbox.

## Prompt Tab

#### Update Prompts
- In the text boxes, edit the content, then click the `Update Prompts` button.

#### Save Prompts
- In the text boxes, edit the content, then click the `Save Prompts` button.

## Config Tab

#### Update Configs
- In the text box, edit the content, then click the `Update Config` button. Note that you can also directly modify the config file `configs/default_release.json` before running the tool. However, update configs through GUI is a convenient way in experimenting with the tool.
- Changing knowledge bases, LLMs etc. can be done through this Tab.

*Selected items in configs:*
- To use another LLM, modify the row of `"model_type": "ehartford/Wizard-Vicuna-13B-Uncensored",`. It is compatible with many models available on Hugging Face.
- To use another Dense Retriever, modify the row of `"encoder": "intfloat/multilingual-e5-base",`. It is compatible with many retrievers available on Hugging Face.
- To limit maximum length of generated response, modify the row of `"max_new_tokens": 512,`.
- To change the number of passages retrieved from knowledge base, modify the row of `"npassage": 5`.

#### Save Configs
- In the text box, edit the content, then click the `Save Config` button
