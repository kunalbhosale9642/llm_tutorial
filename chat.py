import ollama

response = ollama.chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'how are you ?'}]
)

print(response['message']['content'])