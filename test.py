import os
import time
import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
        
        # Load a smaller model (google/flan-t5-base) without quantization,
        # since it's already lightweight enough for testing.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create a text-generation pipeline with the model.
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.1,
            max_new_tokens=256,  # Adjust as needed.
        )
        
        # Wrap the pipeline with LangChain's HuggingFacePipeline.
        self.llm = HuggingFacePipeline(
            pipeline=hf_pipeline,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=False,
        )
        
        # Define a prompt template that instructs the model to base its answer strictly on the provided context.
        self.prompt_template = (
            "[INST] Use the following context strictly to answer the question.\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer clearly and concisely based only on the provided context. "
            "If the context does not contain the answer, state \"I don't know.\" [/INST]"
        )
    
    def query(self, question: str) -> str:
        # Retrieve context from the vector store using similarity search.
        docs = self.vector_db.similarity_search(question, k=5)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Format the prompt.
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Generate the raw response from the model.
        raw_response = self.llm(prompt)
        
        # Post-process the response: remove the prompt portion if present.
        if "[/INST]" in raw_response:
            answer = raw_response.split("[/INST]", 1)[1].strip()
        else:
            answer = raw_response.strip()
        return answer

# Update these paths as needed
vector_db_path = "manit_vector_db/faiss_index"  # Path to your FAISS index
model_name = "google/flan-t5-base"  # Smaller model for testing

# Cache the chatbot so it loads only once.
@st.cache_resource(show_spinner=False)
def load_chatbot():
    return LocalMANITChatbot(vector_db_path, model_name)

chatbot = load_chatbot()

# Streamlit UI
st.title("MANIT Knowledge Assistant - Smaller Model Test")
st.markdown("This app uses the smaller model **google/flan-t5-base** for testing purposes.")

# Text input for user question.
user_input = st.text_input("Your question:", placeholder="Enter your question here...")

if st.button("Submit") and user_input:
    # Display the answer streaming in real-time.
    answer_container = st.empty()
    full_answer = chatbot.query(user_input)
    tokens = full_answer.split()  # Tokenize by whitespace.
    cumulative = ""
    for token in tokens:
        cumulative += token + " "
        answer_container.markdown(cumulative)
        time.sleep(0.05)  # Adjust delay for desired streaming speed.
    st.markdown(f"**Final Answer:** {cumulative}")
