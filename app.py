import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Nepal Tourism Bot", page_icon="🏔️")
st.title("🏔️ Nepal Tourism Guide")
st.caption("Ask me anything about traveling in Nepal. I run 100% locally!")

# --- SIDEBAR ---
with st.sidebar:
    st.header("ℹ️ About this Bot")
    st.markdown("""
    This is a **100% local RAG chatbot**.
    - **Model:** Llama 3.2 3B
    - **Database:** ChromaDB
    - **Framework:** LangChain
    - **UI:** Streamlit
    """)
    
    st.divider()
    st.subheader("💡 Try asking:")
    st.markdown("- *What is the best time to visit?*")
    st.markdown("- *Tell me about Newari food.*")
    st.markdown("- *What should I pack for trekking?*")
    st.markdown("- *Tell me about Chitwan National Park.*")

# --- 2. LOAD THE DATABASE AND LLM ---
@st.cache_resource
def load_components():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = OllamaLLM(model="llama3.2:3b", temperature=0.1, num_predict=500)
    return vectorstore, llm

vectorstore, llm = load_components()

# --- 3. CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. USER INPUT ---
if prompt := st.chat_input("Ask a question about Nepal tourism..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 5. RETRIEVAL & GENERATION ---
    with st.chat_message("assistant"):
        # Search database
        docs = vectorstore.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # BUILD CHAT HISTORY FOR MEMORY
        chat_history = ""
        for msg in st.session_state.messages[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_history += f"{role}: {msg['content']}\n"

        # Build prompt
        full_prompt = f"""You are a helpful Nepal tourism assistant. 
Answer the user's question conversationally using ONLY the context below. 
If the answer is not in the context, say "I don't know based on the provided documents."

Recent Conversation History:
{chat_history}

Context:
{context}

Question: {prompt}
Answer:"""

        # Stream the response
        response = llm.stream(full_prompt)
        answer = st.write_stream(response)

        # Save to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # --- 6. SHOW SOURCES ---
        with st.expander("📚 View Sources"):
            for i, doc in enumerate(docs):
                st.info(f"**Source {i+1}**\n\n{doc.page_content}")
                st.divider()