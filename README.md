# LangChain: Chat with Documents

## Overview
This Streamlit application allows users to upload documents and engage in a conversation about their content using LangChain and OpenAI's language models. The app supports various document formats and provides an intuitive interface for document-based question answering.

## Features
- Upload multiple documents (PDF, DOCX, TXT, EPUB)
- Interactive chat interface for asking questions about the documents
- Utilizes OpenAI's GPT-4 model for generating responses
- Maintains chat history for context-aware conversations
- Document chunking and embedding for efficient retrieval
- Streamlit-based user interface for easy interaction
  

## Setup and Installation
1. Clone this repository:
   ```
   git clone https://github.com/your-username/langchain-chat-with-documents.git
   cd langchain-chat-with-documents
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.streamlit/secrets.toml` file in the project root
   - Add your OpenAI API key to this file:
     ```
     OPENAI_API_KEY = "your-api-key-here"
     ```

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open the provided URL in your web browser.

3. Upload one or more documents using the sidebar.

4. Start asking questions about the uploaded documents in the chat interface.

5. View the chat history and clear it if needed using the sidebar button.

## Project Structure
- `app.py`: Main Streamlit application file
- `chat_with_documents.py`: Contains functions for configuring the retrieval chain
- `utils.py`: Utility functions for document loading and memory management

## How It Works
1. Users upload documents through the Streamlit interface.
2. The app processes the documents, splitting them into chunks and creating embeddings.
3. A retrieval chain is configured using LangChain components.
4. Users can ask questions, which are processed by the retrieval chain.
5. The app uses OpenAI's GPT-4 model to generate responses based on the relevant document chunks.
6. The chat history is maintained for context-aware conversations.

## Customization
- Adjust the `LLM` parameters in `chat_with_documents.py` to change the OpenAI model or its settings.
- Modify the `configure_retriever` function to alter document splitting and retrieval settings.

## Limitations
- The app currently supports a limited number of document formats. Additional formats may require extending the `DocumentLoader` class in `utils.py`.
- Performance may vary depending on the size and complexity of the uploaded documents.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

This application uses OpenAI's language models. Ensure that your usage complies with OpenAI's use-case policies and that you handle any sensitive information appropriately.
