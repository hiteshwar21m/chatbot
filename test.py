import os
import time
import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Define the chatbot class
class LocalMANITChatbot:
    def __init__(self, vector_db_path: str, model_name: str):
        # Initialize embeddings using a SentenceTransformer model.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Load the FAISS vector store from the specified local path.
        self.vector_db = FAISS.load_local(
            vector_db_path, self.embeddings, allow_dangerous_deserialization=True
        )
        
        # Load the smaller model "google/flan-t5-base" using a Seq2Seq model class.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create a text2text-generation pipeline with the model.
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.1,
            max_new_tokens=256,  # Adjust as needed for longer answers.
        )
        
        # Wrap the pipeline with LangChain's HuggingFacePipeline.
        self.llm = HuggingFacePipeline(
            pipeline=hf_pipeline,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=False,
        )
        
        # Define a prompt template that instructs the model to answer strictly based on the provided context.
        self.prompt_template = (
            "[INST] Use the following context strictly to answer the question.\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer clearly and concisely based only on the provided context. "
            "If the context does not contain the answer, state \"I don't know.\" [/INST]"
        )
    
    def query(self, question: str) -> str:
        # Retrieve context documents using similarity search.
        docs = self.vector_db.similarity_search(question, k=5)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Format the prompt with context and question.
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Generate the raw response from the model.
        raw_response = self.llm(prompt)
        
        # Post-process the response to remove the prompt portion if present.
        if "[/INST]" in raw_response:
            answer = raw_response.split("[/INST]", 1)[1].strip()
        else:
            answer = raw_response.strip()
        return answer

# Update these paths as needed.
vector_db_path = "manit_vector_db/faiss_index"  # Path to your FAISS index
model_name = "google/flan-t5-base"  # Smaller model for testing

# Cache the chatbot so it loads only once.
@st.cache_resource(show_spinner=False)
def load_chatbot():
    return LocalMANITChatbot(vector_db_path, model_name)

chatbot = load_chatbot()

# Define a streaming function that yields the answer word-by-word.
def chatbot_interface_stream(message):
    try:
        full_answer = chatbot.query(message)
        tokens = full_answer.split()  # Tokenize by whitespace.
        cumulative = ""
        for token in tokens:
            cumulative += token + " "
            yield cumulative  # Yield the cumulative answer.
            time.sleep(0.05)  # Adjust delay as needed.
    except Exception as e:
        yield f"Error processing request: {str(e)}"

# Build a ChatGPT-like interface using Streamlit.
st.title("MANIT Knowledge Assistant - Smaller Model Test")
st.markdown("This app uses **google/flan-t5-base** for testing. It streams the answer word-by-word.")

# Create a text input for the question.
user_input = st.text_input("Your question:", placeholder="Enter your question here...")

if st.button("Submit") and user_input:
    # Display the streaming answer.
    answer_placeholder = st.empty()
    # Use the streaming generator to update the answer.
    for partial in chatbot_interface_stream(user_input):
        answer_placeholder.markdown(partial)
