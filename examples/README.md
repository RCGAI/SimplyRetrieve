## API Access
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
