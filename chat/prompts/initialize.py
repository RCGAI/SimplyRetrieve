import importlib

# Initialize prompts
def initialize_prompts(kwargs):
    for k, v in kwargs.items():
        if k == 'prompt_config':
            for kk, vv in v.items():
                if isinstance(vv, str) and vv.startswith('prompts.'):
                    prompt_module = importlib.import_module('.'.join(vv.split('.')[:-2]))
                    prompt_class = getattr(prompt_module, vv.split('.')[-2])()
                    prompt = getattr(prompt_class, vv.split('.')[-1])
                    kwargs[k][kk] = prompt
    return kwargs

# Save Prompts Function
def save_prompts(*prompts):
    with open("./prompts/PREFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[0] == '':
            writer.write(prompts[0])
        elif prompts[0][-1] == '\n':
            writer.write(prompts[0]+'\n')
        else:
            writer.write(prompts[0])

    with open("./prompts/SUFFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[1] == '':
            writer.write(prompts[1])
        elif prompts[1][-1] == '\n':
            writer.write(prompts[1]+'\n')
        else:
            writer.write(prompts[1])

    with open("./prompts/RETRIEVEPREFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[2] == '':
            writer.write(prompts[2])
        elif prompts[2][-1] == '\n':
            writer.write(prompts[2]+'\n')
        else:
            writer.write(prompts[2])

    with open("./prompts/RETRIEVESUFFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[3] == '':
            writer.write(prompts[3])
        elif prompts[3][-1] == '\n':
            writer.write(prompts[3]+'\n')
        else:
            writer.write(prompts[3])

    with open("./prompts/AIPREFIXBASIC_SAVED.txt", "w", encoding="utf-8") as writer:
        if prompts[4] == '':
            writer.write(prompts[4])
        elif prompts[4][-1] == '\n':
            writer.write(prompts[4]+'\n')
        else:
            writer.write(prompts[4])

    print("Prompts save completed")
