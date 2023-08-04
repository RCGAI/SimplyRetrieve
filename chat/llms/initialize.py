# Copyright (c) Kioxia Corporation and its affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
import torch

def initialize_llm(model_args={}, pipeline_args={}):

    quantization_module = importlib.import_module('.'.join(model_args['quantization_config']['quantization_type'].split('.')[:-1]))
    quantization_class = getattr(quantization_module, model_args['quantization_config']['quantization_type'].split('.')[-1])
    quantization_config_config = model_args['quantization_config']['quantization_config_config']
    quantization_config = quantization_class(**quantization_config_config)

    print('Loading Tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_args['model_type'])
    print("special tokens used: ", tokenizer.all_special_tokens, " id: ", tokenizer.all_special_ids)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model_kwargs = model_args.copy()
    del model_kwargs['model_type'], model_kwargs['device_map'], model_kwargs['quantization_config']
    print('Loading LLM model...')
    model = AutoModelForCausalLM.from_pretrained(
                model_args['model_type'],
                device_map=model_args['device_map'],
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                **model_kwargs)

    model.eval()

    pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                streamer=streamer,
                **pipeline_args)

    def generate_fn(prompts, **kwargs):

        if isinstance(prompts, str):
            return pipe([prompts], **kwargs)[0][0]["generated_text"][len(prompts) :]
        elif isinstance(prompts, list):
            outputs = pipe(prompts, **kwargs)
            results = [out[0]["generated_text"][len(prompt) :] for prompt, out in zip(prompts, outputs)]
            return results
        else:
            raise ValueError

    main_agent_llm = generate_fn

    return main_agent_llm, streamer
