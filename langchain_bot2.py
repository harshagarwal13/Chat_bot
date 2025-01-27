import os
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "test_3")

# Define the embedding model
embeddings = OllamaEmbeddings(model="llama3.2")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)
response = ""
while True:
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = f"{response}. Human message is: {query}"
# Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
#     combined_input = (
#         "If the user is having normal conversation then interact with the user and if there is some question regarding something then here are some documents that might help answer the question: "
#         + response
#         + "\n\nRelevant Documents:\n"
#         + "\n\n".join([doc.page_content for doc in relevant_docs])
#         + "\n\nPlease provide an answer based only on the provided documents. Also the last user message is updated, so answer accordingly and don't answer solely on previous messages and take out result from the document '"
#     )
#     combined_input = (
#             "If the user is having normal or casual conversation then interact with the user solely based on that and if there is some question regarding something related to document then here are some documents that might help answer the question: "
#             + response
#             + "\n\nRelevant Documents:\n"
#             + "\n\n".join([doc.page_content for doc in relevant_docs])
#             + "\n\nPlease provide an answer based only on the provided documents. Also the last Human message should be updated, so answer accordingly and do not answer solely on other previous messages and take out result from the document if not required Don't respond out anything meaningless'"
#     )
#     combined_input = (
#         f"If the user is engaging in a general conversation, interact naturally but keep it short and behave like a bot. If the user asks a question or seeks specific information, here are some documents that might help answer the query."
#         + response
#         + ":(Provide an answer based only on the provided documents.)"
#         +"Relevant Documents: Include relevant sections of the documents for context."
#         + "\n".join([doc.page_content for doc in relevant_docs])
#         # + "Ensure your answer is based solely on the content from the provided documents.Avoid relying on prior conversation history unless it is part of the current user message or stated as context.Always prioritize user - provided material."
#     )
    combined_input = (
        "You are a helpful bot to give out information. Have professional conversation with the user and help the user by giving answer to what they ask by referencing the provided data below"
        + response
        + ":(Provide an answer based only on the provided data.)"
        + "\n".join([doc.page_content for doc in relevant_docs])
        + "Also Avoid relying on prior conversation history unless it is part of the current user message or stated as context.Always prioritize user - provided material."
        # + "Don't say that you have any document or something from where you have extracted the data"
        # + "Let me give you example of the prior conversation the user response will be as 'user message is' and your prior response will be like 'your message is' "
    )
    model = OllamaLLM(model="llama3.2")
    messages = [
        SystemMessage(content="You are a helpful chatbot."),
        HumanMessage(content=combined_input),
    ]

    # Invoke the model with the combined input
    result = model.invoke(messages)
    print(result)
    response = f"{response} your message is: {result}"

