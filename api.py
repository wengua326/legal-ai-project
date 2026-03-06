import os
from openai import OpenAI

client = OpenAI(
    base_url = "https://api.openai.com/v1",
    api_key = "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"
)

messages = [
        {
            "role": "system",
            "content": "You are a professional legal counsel. Provide Malaysia legal info from a neutral perspective. Dont give any legal advice."
        }
    ]

print("Conversation start! (enter 'exit' to quit)")

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Conversation ended.")
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
    model = "openai/gpt-oss-20b",
    messages = messages,
    max_tokens = 1000,
    temperature = 0.1,
    stream = True
    )

    for chunk in response:
        if chunk.choices[0].delta.content :
            print("AI: " + chunk.choices[0].delta.content, end="", flush=True)
            full_response += content
    
    messages.append({"role": "assistant", "content": full_response})
    print()