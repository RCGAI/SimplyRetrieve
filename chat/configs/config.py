# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

# Read Config File Function
def read_config(path):
    with open(path, "r", encoding="utf-8") as reader:
        text = reader.read()
    loaded = json.loads(text)
    return loaded

# Write Config File Function
def save_config(config, path):
    json_dict = json.loads(config)
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(json_dict, writer, ensure_ascii=False, indent=4)
    print("Save Config Completed")
    return "Save Complete"
