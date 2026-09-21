from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:3b")

response = llm.invoke("What is the capital of Nepal?")

print(response)