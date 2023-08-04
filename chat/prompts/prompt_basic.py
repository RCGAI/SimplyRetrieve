# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pydantic import BaseModel

class PromptBasic(BaseModel):
    EMPTY: str = ""

    PREFIXBASIC: str = "Human:"

    SUFFIXBASIC: str = """
AI:"""

    RETRIEVEPREFIXBASIC: str ='"'

    RETRIEVESUFFIXBASIC: str ='''"
answer the following question with the provided knowledge.
'''

class PromptExternal():
    EMPTY: str = ""

    with open("./prompts/PREFIXBASIC.txt", "r", encoding="utf-8") as reader:
        text = reader.read()
        PREFIXBASIC: str = text[:-1]

    with open("./prompts/SUFFIXBASIC.txt", "r", encoding="utf-8") as reader:
        text = reader.read()
        SUFFIXBASIC: str = text[:-1]

    with open("./prompts/RETRIEVEPREFIXBASIC.txt", "r", encoding="utf-8") as reader:
        text = reader.read()
        RETRIEVEPREFIXBASIC: str = text[:-1]

    with open("./prompts/RETRIEVESUFFIXBASIC.txt", "r", encoding="utf-8") as reader:
        text = reader.read()
        RETRIEVESUFFIXBASIC: str = text[:-1]
