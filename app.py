import logging

import streamlit as st
from streamlit.external.langchain import StreamlitCallbackHandler

from chat_with_documents import configure_retrieval_chain
from utils import MEMORY, DocumentLoader



logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER = logging.getLogger()

st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.header("🦜 LangChain: Chat with Documents")

# Add instructions on how to use the app
st.markdown("""
## Instructions
1. **Upload Documents**: Use the sidebar to upload one or multiple documents. Supported formats include PDF, DOCX, TXT, and more.
2. **Interact with the Documents**: After uploading, ask any questions related to the content of the documents.
3. **View Chat History**: The chat history is maintained, and you can clear it anytime using the sidebar button.

### Example Questions:
- "Summarize the key points from the uploaded documents."
- "What are the main findings in the first document?"
- "Can you provide an overview of the uploaded research papers?"

Feel free to experiment and ask anything related to the content of the uploaded documents!
""")

uploaded_files = st.sidebar.file_uploader(
    label="Upload files",
    type=list(DocumentLoader.supported_extensions.keys()),
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

CONV_CHAIN = configure_retrieval_chain(uploaded_files)

if st.sidebar.button("Clear message history"):
    MEMORY.chat_memory.clear()
    st.experimental_rerun()

avatars = {"human": "user", "ai": "assistant"}

if len(MEMORY.chat_memory.messages) == 0:
    st.chat_message("assistant").markdown("Ask me anything!")

for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

assistant = st.chat_message("assistant")

if user_query := st.chat_input(placeholder="Ask a question related to the uploaded documents"):
    st.chat_message("user").write(user_query)
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)
    with st.chat_message("assistant"):
        params = {
            "question": user_query,
            "chat_history": MEMORY.chat_memory.messages,
        }
        response = CONV_CHAIN.run(params, callbacks=[stream_handler])
        # Display the response from the chatbot
        if response:
            container.markdown(response)
