import ollama
import sys

# https://ollama.com/download/

# To settup on Windows, follow:
# https://www.ralgar.one/ollama-on-windows-a-beginners-guide/

# Ollama installation on Linux (and Mac):
# curl -fsSL https://ollama.com/install.sh | sh
# pip install ollama

# Select suitable model. Larger is better, but requires more memory
# https://machinelearningmastery.com/top-7-small-language-models-you-can-run-on-a-laptop/

# Define the model

# This one is smallest
# Requires less than 1GB memory
# https://ollama.com/library/qwen2.5
# ollama pull qwen3:0.6b
model = 'qwen3:0.6b'

# Requires 2-4GB RAM
# ollama pull llama3.2:1b
# model = 'llama3.2:1b'

# https://ollama.com/library/ministral-3
# ollama pull ministral-3:8b
# model = 'ministral-3:8b'
# Requires 10GB RAM 

# Even smaller:
# https://ollama.com/library/smollm2

# Badly written text to be improved:
prompt = '"In conclushion, while havving a broad knowledge of many academmic subjects can be beneficial in certain situations, I believe that specializing in one subject is often more beneficial. Specializing in one subject can help someone to becom an expert in the field, to develop a deeep understandi ng of the subject, and to develop a pashion for the subject. For these raisons, I bellieve that specializing in on subject is often more beneficial thann having a broadd knowwledge of many academic subjects."'
# prompt = sys.argv[1]

full_prompt = (
    "You are an AI that improves grammar and text flow of given paragraphs. "
    "Return a concise improved version of the text. "
    "Do not include any extra information.\n\n"
    f"{prompt}"
)
response = ollama.generate(
    model=model,
    prompt=full_prompt,
    think=False, # May need to comment this out for other models
    options={
        'temperature': 1.0,
        'num_predict': 500,
        'top_k': 200,
        'top_p': 0.90,
        'min_p': 0.0001,
        # 'seed': 42,
    }
)

# import pdb ;breakpoint()
print(response['response'])



