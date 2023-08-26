# GUI Operation Manual of SimplyRetrieve
This manual provides simple instructions on operating the GUI of this tool. The manual will be updated accordingly.

## Knowledge Tab
- `Drag and drop` new documents (pdf etc.), then click the `Create Knowledge` button to create knowledge base on-the-fly directly from GUI. Support multiple documents.
- If same Knowledgebase Filename existed, new knowledge will be automatically appended to that knowledge base.
- After knowledge base creation, Chat Tab will be automatically updated to include the new knowledge. New knowledge can be selected from the dropdown menu of knowledgebase under Chat Tab. For appended knowledge, current knowledge base used will be automatically updated.
- To make the newly created knowledge base permanent even after reloading the tool, click the `Save Config` button under Config Tab to save a copy of the automatically updated configs. Specify the new config file in command line the next time you run the tool.

## Chat Tab

#### Ask Question to Get a Response
- In the text box, type in your query (question for Generative AI tool) and press `Enter`.

#### Enable/Disable Knowledge Base
- Check or uncheck the `Use Knowledgebase` checkbox.

#### Enable/Disable Mixture-of Knowledge-Base
- Under `KnowledgeBase Mode`, select the mode from the pulldown menu.
- This function will only work when you attach more than 1 Knowledge Base to the tool.

#### Enable/Disable Explicit Prompt-Weighting
- Check or uncheck the `Prompt-Weighting` checkbox.
- Adjust the slider to increase or decrease the weight.

## Prompt Tab

#### Update Prompts
- In the text boxes, edit the content, then click the `Update Prompts` button.

#### Save Prompts
- In the text boxes, edit the content, then click the `Save Prompts` button.
- Prompts will be saved in the subdirectory of `prompts/` in separate files.

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
- In the text box, edit the content, then click the `Save Config` button.
- Config will be saved in the subdirectory of `configs/` in separate file. You can change the `Save Path` next to the Save Config button.

## Analysis Tab
- Click the `Load Logs` button to display the query-response history and analytical info.
- Click the `Save Logs` button to save the query-response history and analytical info.
- Logs will be saved in the subdirectory of `analysis/`. You can change the `Save Log Path` next to the Save Logs button.
