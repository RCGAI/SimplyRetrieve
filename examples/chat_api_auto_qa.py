# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from gradio_client import Client
import json
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='question_sample1.json')
parser.add_argument('--output', type=str, default='answer_sample1.json')
parser.add_argument('--ipaddr', type=str, default='localhost')
parser.add_argument('--port', type=str, default='7860')
args = parser.parse_args()

with open(args.input, "r", encoding="utf-8") as f:
    questions = json.load(f)

print("number of questions:", len(questions))
print("input file loaded")

client = Client(f"http://{args.ipaddr}:{args.port}/")
print("IP address set")

print("start querying")

t0 = datetime.datetime.now()

for i, q in enumerate(questions):

    try:
        result = client.predict(
            q["question"],
            True,	                # Use KnowledgeBase
            "Selectable KnowledgeBase",	# KnowledgeBase Mode
            "Kioxia Expert",	        # KnowledgeBase Selection
            True,	                # Use Prompt-Weighting
            100,                        # KnowledgeBase Weightage
	    True,	                # Enable Data Logging
            api_name="/message-query"
        )
        
        q["answer"] = result
        print(f"\n{q}")
        
    except:
        # q["answer"] = ""
        print("\nfailed")

    if i%10==0:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=0)

with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=0)

print(f"\nprocessing time: {datetime.datetime.now() - t0}, \
      time/query: {(datetime.datetime.now() - t0).total_seconds()/len(questions):.2f} sec")
print("all queries processed")
