from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b", temperature=0.1, num_predict= 80)

response = llm.stream("Tell me a short story about a robot learning to cook.")

for chunk in response:
    print(chunk, end="", flush=True)