import ollama

response = ollama.chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'What are the main features of Gemma 3?'}]
)

print(response['message']['content'])