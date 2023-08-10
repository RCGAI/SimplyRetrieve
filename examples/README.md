## API Access
This is the manual for API access to the tool. You will need to run the tool first before accessing it through API. API examples below not only allow query through terminal interface, but also can be used for large-scale experiments and benchmarkings through batched access.

### Single Query Access
Run the script below for single query API access. Replace `Where is Penang` with your own query.
```
python chat_api_access.py --query "Where is Penang?"
```
### Batched Queries Access
Run the script below for batched API access of queries. Replace `question_sample1.json` with your own json file of queries. Response will be saved automatically in output json file.
```
python chat_api_auto_qa.py --input question_sample1.json --output answer.json
```
Run the script below for batched API access of queries and response evaluations. Replace `question_sample2.json` with your own json file of queries and labels. Response will be saved automatically in output json file. We utilize the *evaluate* library by Hugging Face for Rouge scores calculation. Inference time per query is shown on the terminal too after running the script.
```
python chat_api_auto_eval.py --input question_sample2.json --output answer.json --knowledge 1
```
