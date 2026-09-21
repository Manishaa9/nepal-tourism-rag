from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b", temperature=0.1, num_predict=150)

print("Khalti Bot is ready! Type 'quit' to exit.")

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    print("Bot: ", end="")
    response = llm.stream(user_input)
    for chunk in response:
        print(chunk, end="", flush=True)