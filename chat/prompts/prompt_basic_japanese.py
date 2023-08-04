# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pydantic import BaseModel

class PromptBasic(BaseModel):
    EMPTY: str = ""
    PREFIXBASIC: str = "質問："

    SUFFIXBASIC: str = """
回答："""

    RETRIEVEPREFIXBASIC: str ='"'

    RETRIEVESUFFIXBASIC: str ='''"
上記知識を参考にして回答してください。
'''

class PromptExternal():
    EMPTY: str = ""

    with open("./prompts/PREFIXBASIC_JPN.txt", "r", encoding="utf-8") as reader:
        text = reader.read()
        PREFIXBASIC: str = text[:-1]

    with open("./prompts/SUFFIXBASIC_JPN.txt", "r", encoding="utf-8") as reader:
        text = reader.read()
        SUFFIXBASIC: str = text[:-1]

    with open("./prompts/RETRIEVEPREFIXBASIC_JPN.txt", "r", encoding="utf-8") as reader:
        text = reader.read()
        RETRIEVEPREFIXBASIC: str = text[:-1]

    with open("./prompts/RETRIEVESUFFIXBASIC_JPN.txt", "r", encoding="utf-8") as reader:
        text = reader.read()
        RETRIEVESUFFIXBASIC: str = text[:-1]
