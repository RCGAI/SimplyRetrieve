## API Access
This is the manual for API access to the tool. You will need to run the tool first before accessing it through API. API examples below not only allow query through terminal interface, but also can be use for large-scale experiments and benchmarkings through batched access.

Run the script below for single query API access.
```
python chat_api_access.py --query "Where is Penang?"
```
Run the script below for batched API access of queries. Please edit `question_sample1.json` to create your own queries.
```
python chat_api_auto_qa.py --input question_sample1.json
```
Run the script below for batched API access of queries and response evaluations. Please edit `question_sample2.json` to create your own queries.
```
python chat_api_auto_eval.py --input question_sample2.json
```
