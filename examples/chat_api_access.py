# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from gradio_client import Client
import json
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, default='Where is Penang?')
parser.add_argument('--ipaddr', type=str, default='localhost')
parser.add_argument('--port', type=str, default='7860')
args = parser.parse_args()

client = Client(f"http://{args.ipaddr}:{args.port}/")
print("IP address set")

print("start querying")

t0 = datetime.datetime.now()

try:
    result = client.predict(
        args.query,
        True,	                        # Use KnowledgeBase
        "Selectable KnowledgeBase",	# KnowledgeBase Mode
        "Kioxia Expert",	        # KnowledgeBase Selection
        True,	                        # Use Prompt-Weighting
        100,                            # KnowledgeBase Weightage
        True,	                        # Enable Data Logging
        api_name="/message-query"
    )
    print(f"\n{result}")
        
except:
    print("\nfailed")

print(f"\nprocessing time: {datetime.datetime.now() - t0}")
