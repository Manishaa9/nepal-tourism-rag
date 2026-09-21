from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

# 1. Connect to our existing database
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Set up the LLM
llm = OllamaLLM(model="llama3.2:3b", temperature=0.1)

print("Nepal Tourism Bot is ready! Type 'quit' to exit.")

# 3. Start the chat loop
while True:
    question = input("\nAsk a question about Nepal tourism: ")
    
    if question.lower() == "quit":
        print("Goodbye!")
        break

    # 4. Search the database
    print("\n--- Searching database ---")
    docs = vectorstore.similarity_search(question, k=2)

    # Show found chunks
    for i, doc in enumerate(docs):
        print(f"Found Chunk {i+1}: {doc.page_content[:100]}...") # Just show a preview

    # 5. Build the prompt
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Improved prompt to make it more conversational
    prompt = f"""You are a helpful Nepal tourism assistant. 
Answer the user's question conversationally using ONLY the context below. 
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""

    # 6. Stream the answer
    print("\n--- Bot Answer ---")
    response = llm.stream(prompt)
    for chunk in response:
        print(chunk, end="", flush=True)
    print("\n")