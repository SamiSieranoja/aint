import ollama
import sys

# Define the model
model = 'llama3.2:1b' # Parempi, toimii ainakin jos koneessa 16GB muistia
# model = 'qwen:0.5b' # Nopeampi, vaatii vähemmän muistia

# Ollama asennus Linuxissa:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull 'qwen:0.5b' 
# ollama pull 'llama3.2:1b' 
# pip install ollama

prompt = '"In conclushion, while havving a broad knowledge of many academmic subjects can be beneficial in certain situations, I believe that specializing in one subject is often more beneficial. Specializing in one subject can help someone to becom an expert in the field, to develop a deeep understandi ng of the subject, and to develop a pashion for the subject. For these raisons, I bellieve that specializing in on subject is often more beneficial thann having a broadd knowwledge of many academic subjects."'
# prompt = sys.argv[1]

for i in range(0,5):
	response = ollama.chat(
		model=model,
		messages=[
		{
		'role': 'system',
		'content': 'You are an AI that improves grammar and text flow of given paragraphs. The user will provide a paragraph of text. Return a concise improved version of that text. Do not include any extra information.',
	
		},   
		{'role': 'user', 'content': prompt}],
		options={'temperature': 1.0, 'num_predict': 100, 'top_k':200,
		# "num_keep": 5,
		# "seed": 42,
		"top_p": 0.99,
		"min_p": 0.0001,
		# "typical_p": 0.7,
		# "repeat_last_n": 33,
		# "repeat_penalty": 1.2,
		# "presence_penalty": 1.5,
		# "frequency_penalty": 1.0,
		# "mirostat": 1,
		# "mirostat_tau": 0.8,
		# "mirostat_eta": 0.6,
		# "penalize_newline": true,
		# "stop": ["\n", "user:"],
		# "numa": false,
		# "num_ctx": 1024,
		# "num_batch": 2,
		# "num_gpu": 1,
		# "main_gpu": 0,
		# "low_vram": false,
		# "vocab_only": false,
		# "use_mmap": true,
		# "use_mlock": false,
		# "num_thread": 8
	}
	)
	print(f"◼ {i} {response['message']['content']}\n")


