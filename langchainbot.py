# import os
# from langchain.text_splitter import (
#     CharacterTextSplitter,
#     RecursiveCharacterTextSplitter,
#     SentenceTransformersTokenTextSplitter,
#     TextSplitter,
#     TokenTextSplitter,
# )
# from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# # from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir,"hr_manual.txt")
# persistent_directory = os.path.join(current_dir, "db", "test_3")
# # Check if the Chroma vector store already exists
# if not os.path.exists(persistent_directory):
#     print("Persistent directory does not exist. Initializing vector store...")
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(
#             f"The file {file_path} does not exist. Please check the path."
#         )
#
#     # Read the text content from the file
#     loader = TextLoader(file_path)
#     documents = loader.load()
#     # Split the document into chunks
#     class CustomTextSplitter(TextSplitter):
#         def split_text(self, text):
#             # Custom logic for splitting text
#             return text.split("\n\n")  # Example: split by paragraphs
#
#
#     custom_splitter = CustomTextSplitter()
#     custom_docs = custom_splitter.split_documents(documents)
#     # create_vector_store(custom_docs, "chroma_db_custom")
#     # rec_char_splitter = RecursiveCharacterTextSplitter()
#     # rec_char_docs = rec_char_splitter.split_documents(documents)
#     # text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
#     # rec_char_docs = text_splitter.split_documents(documents)
# # if not os.path.exists(persistent_directory):
# #     print("Persistent directory does not exist. Initializing vector store...")
# #
# #     # Ensure the books directory exists
# #     if not os.path.exists(books_dir):
# #         raise FileNotFoundError(
# #             f"The directory {books_dir} does not exist. Please check the path."
# #         )
# #
# #     # List all text files in the directory
# #     book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
# #
# #     # Read the text content from each file and store it with metadata
# #     documents = []
# #     for book_file in book_files:
# #         file_path = os.path.join(books_dir, book_file)
# #         loader = TextLoader(file_path)
# #         book_docs = loader.load()
# #         for doc in book_docs:
# #             # Add metadata to each document indicating its source
# #             doc.metadata = {"source": book_file}
# #             documents.append(doc)
# #
# #     # Split the documents into chunks
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# #     docs = text_splitter.split_documents(documents)
#     # Display information about the split documents
#     # print("\n--- Document Chunks Information ---")
#     # print(f"Number of document chunks: {len(docs)}")
#     # print(f"Sample chunk:\n{docs[0].page_content}\n")
#
#     # Create embeddings
#     print("\n--- Creating embeddings ---")
#     # embeddings = OpenAIEmbeddings(
#     #     model="text-embedding-3-small"
#     # )  # Update to a valid embedding model if needed
#     # print(rec_char_docs)
#     embeddings = OllamaEmbeddings(
#         model="llama3.2",
#     )
#     # sample_embeddings = embeddings.embed_documents([doc.page_content for doc in rec_char_docs[:5]])
#     # print(sample_embeddings)
#     print("\n--- Finished creating embeddings ---")
#
#     # Create the vector store and persist it automatically
#     print("\n--- Creating vector store ---")
#     db = Chroma.from_documents(
#         custom_docs, embeddings, persist_directory=persistent_directory)
#     print("\n--- Finished creating vector store ---")
#
# else:
#     print("Vector store already exists. No need to initialize.")


# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_ollama import OllamaLLM
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, "hr_manual.txt")
# persistent_directory = os.path.join(current_dir, "db", "improved_db2")
#
# # Improved document processing
# if not os.path.exists(persistent_directory):
#     print("Initializing vector store...")
#     loader = TextLoader(file_path)
#     documents = loader.load()
#
#     # Better text splitting with context preservation
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         # separators=["\n\n", "\n", "\. ", " ", ""]
#     )
#     splits = text_splitter.split_documents(documents)
#
#     # Initialize embeddings
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#
#     # Create vector store
#     db = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory=persistent_directory
#     )
# else:
#     db = Chroma(persist_directory=persistent_directory,
#                 embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
#
# # Improved retrieval configuration
# retriever = db.as_retriever(
#     search_type="mmr",  # Maximal marginal relevance for better diversity
#     search_kwargs={
#         "k": 6,
#         "score_threshold": 0.4
#     }
# )
#
# # Enhanced prompt template
# template = """You are an HR assistant for our company. Follow these rules:
# 1. Answer ONLY using the context provided
# 2. For HR policy questions, be precise
# 3. For non-HR questions, respond politely that you specialize in HR policies
# 4. If unsure, say "I need to verify that information. Please check with HR directly."
#
# Context: {context}
# Question: {question}
# """
#
# prompt = ChatPromptTemplate.from_template(template)
# llm = OllamaLLM(model="llama3.2")
#
# # Chain setup
# rag_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
# )
#
# # Interaction loop
# print("HR Assistant initialized. Ask about company policies or type 'exit'")
# # arr = []
# # string = ','.join(str(x) for x in arr)
# while True:
#     query = input("\nYour question: ").strip()
#     if query.lower() == 'exit':
#         break
#     # arr.append(query)
#     # if len(arr)>5:
#     #     arr = arr[1:]
#     # res = '.'.join(str(x) for x in arr)
#     # print("-----------------------------"+res+"-------------------------")
#     response = rag_chain.invoke(query)
#     print("\nAssistant:", response)


# import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_ollama import OllamaLLM
# from langchain_core.messages import AIMessage, HumanMessage
# from typing import List, Tuple
#
# current_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(current_dir, "hr_manual.txt")
# persistent_directory = os.path.join(current_dir, "db", "improved_db")
#
# # Initialize or load vector store
# if not os.path.exists(persistent_directory):
#     print("Initializing vector store...")
#     loader = TextLoader(file_path)
#     documents = loader.load()
#
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", "\. ", " ", ""]
#     )
#     splits = text_splitter.split_documents(documents)
#
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     db = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory=persistent_directory
#     )
# else:
#     db = Chroma(persist_directory=persistent_directory,
#                 embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
#
#
# # Configure retriever with history-aware search
# class HistoryAwareRetriever:
#     def __init__(self, base_retriever):
#         self.base_retriever = base_retriever
#
#     def format_history(self, history: List[Tuple[str, str]]) -> str:
#         """Convert chat history into contextual string"""
#         context_str = ""
#         for human, ai in history[-3:]:  # Keep last 3 exchanges
#             context_str += f"Previous Question: {human}\nPrevious Answer: {ai}\n\n"
#         return context_str.strip()
#
#     def get_relevant_documents(self, query: str, history: List[Tuple[str, str]] = []):
#         """Enhance query with conversation history"""
#         historical_context = self.format_history(history)
#         enhanced_query = f"{historical_context}\nCurrent Question: {query}"
#         return self.base_retriever.get_relevant_documents(enhanced_query)
#
#
# # Create retriever with history support
# base_retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 6, "score_threshold": 0.4}
# )
# retriever = HistoryAwareRetriever(base_retriever)
#
# # Enhanced prompt template with history support
# template = """You are an HR assistant. Use this conversation history and context to answer:
# {history}
#
# Context: {context}
# Current Question: {question}
#
# Follow these rules:
# 1. Answer using only the context and history
# 2. Maintain consistent policy interpretations
# 3. Acknowledge follow-ups explicitly ("Regarding your previous question...")
# 4. For unknown queries: "I need to verify that. Please check with HR directly."
# """
#
# prompt = ChatPromptTemplate.from_template(template)
# llm = OllamaLLM(model="llama3.2")
#
#
# # Chain setup with history
# def format_docs(docs):
#     return "\n\n".join(d.page_content for d in docs)
#
#
# rag_chain = (
#         RunnablePassthrough.assign(
#             context=lambda x: format_docs(retriever.get_relevant_documents(x["question"], x["history"])),
#             question=lambda x: x["question"]
#         )
#         | prompt
#         | llm
# )
#
#
# # Conversation manager
# class ConversationManager:
#     def __init__(self):
#         self.history = []
#
#     def add_exchange(self, question: str, answer: str):
#         """Store conversation history"""
#         self.history.append((question, answer))
#         # Keep only last 5 exchanges
#         if len(self.history) > 5:
#             self.history = self.history[-5:]
#
#     def get_history(self):
#         """Get formatted history string"""
#         return "\n".join(
#             [f"Human: {q}\nAssistant: {a}" for q, a in self.history]
#         )
#
#
# # Interaction loop
# print("HR Assistant initialized. Ask about company policies or type 'exit'")
# conv_manager = ConversationManager()
#
# while True:
#     query = input("\nYour question: ").strip()
#     if query.lower() == 'exit':
#         break
#
#     # Prepare inputs with history
#     inputs = {
#         "question": query,
#         "history": conv_manager.get_history()
#     }
#
#     # Generate response
#     response = rag_chain.invoke(inputs)
#
#     # Store conversation
#     conv_manager.add_exchange(query, response)
#
#     print("\nAssistant:", response)


import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from typing import List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "hr_manual.txt")
persistent_directory = os.path.join(current_dir, "db", "improved_db")

# Initialize vector store (same as before)
if not os.path.exists(persistent_directory):
    print("Initializing vector store...")
    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "\. ", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persistent_directory
    )
else:
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

# Fixed HistoryAwareRetriever implementation
class HistoryAwareRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def format_history(self, history: List[Tuple[str, str]]) -> str:
        """Convert chat history into contextual string"""
        context_str = ""
        # Handle empty history case
        if not history:
            return ""

        # Ensure we only take the last 3 exchanges
        for exchange in history[-3:]:
            # Verify exchange format
            if len(exchange) != 2:
                continue
            human, ai = exchange
            context_str += f"Previous Question: {human}\nPrevious Answer: {ai}\n\n"
        return context_str.strip()

    def get_relevant_documents(self, query: str, history: List[Tuple[str, str]] = []):
        """Enhanced query with conversation history"""
        historical_context = self.format_history(history)
        enhanced_query = f"{historical_context}\nCurrent Question: {query}"
        return self.base_retriever.get_relevant_documents(enhanced_query) #Changes made here


# Create retriever with history support
base_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "score_threshold": 0.4}
)
retriever = HistoryAwareRetriever(base_retriever)

# Enhanced prompt template with history support
template = """You are an HR assistant. Use this conversation history and context to answer:
{formatted_history}

Context: {context}
Current Question: {question}

Follow these rules:
1. Answer using only the context and history
2. Maintain consistent policy interpretations
3. Acknowledge follow-ups explicitly ("Regarding your previous question...")
4. For unknown queries: "I need to verify that. Please check with HR directly."
5. For non-HR questions, respond politely that you specialize in HR policies
"""

prompt = ChatPromptTemplate.from_template(template)
llm = OllamaLLM(model="llama3.2")


# Chain setup with history
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


class ConversationManager:
    def __init__(self):
        self.history = []

    def add_exchange(self, question: str, answer: str):
        """Store conversation history as tuples"""
        self.history.append((question, answer))
        # Keep only last 3 exchanges to match retriever context
        if len(self.history) > 3:
            self.history = self.history[-3:]

    def get_formatted_history(self):
        """Format history for prompt"""
        return "\n".join(
            [f"Human: {q}\nAssistant: {a}" for q, a in self.history]
        )


# Interaction loop
print("HR Assistant initialized. Ask about company policies or type 'exit'")
conv_manager = ConversationManager()

while True:
    query = input("\nYour question: ").strip()
    if query.lower() == 'exit':
        break

    try:
        # Get relevant documents using conversation history
        docs = retriever.get_relevant_documents(query, conv_manager.history)
        # print(docs)
        # Format inputs for the prompt
        inputs = {
            "formatted_history": conv_manager.get_formatted_history(),
            "context": format_docs(docs),
            "question": query
        }

        # Generate response
        response = prompt.invoke(inputs)
        response = llm.invoke(response)

        # Store conversation
        conv_manager.add_exchange(query, response)

        print("\nAssistant:", response)

    except Exception as e:
        print(f"\nError processing request: {str(e)}")
