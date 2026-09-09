from ollama import chat
prompt = "The capital of India is"
response = chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
print("Prompt:", prompt)
print("Model response:", response.message.content)