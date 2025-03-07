import os
import time
import traceback
import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# If running on Streamlit Cloud, set your Hugging Face token from secrets.
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

class LocalMANITChatbot:
    def __init__(self, vector_db_path: str, model_name: str):
        # Initialize embeddings using a SentenceTransformer model.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Load the FAISS vector store from the specified path.
        self.vector_db = FAISS.load_local(
            vector_db_path, self.embeddings, allow_dangerous_deserialization=True
        )
        
        # Load the quantized model using Transformers with bitsandbytes support in 4-bit precision.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,       # Enable 4-bit quantization.
            device_map="auto"        # Automatically assign the model to available devices.
        )
        
        # Create a text-generation pipeline with the quantized model.
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.1,
            max_new_tokens=512,  # Adjust as needed.
        )
        
        # Wrap the pipeline with LangChain's HuggingFacePipeline.
        self.llm = HuggingFacePipeline(
            pipeline=hf_pipeline,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=False,
        )
        
        # Define a custom prompt template that instructs the model to answer strictly based on the provided context.
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
        
        # Format the prompt with the retrieved context and question.
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Generate the raw response from the quantized model.
        raw_response = self.llm(prompt)
        
        # Post-process: remove the prompt portion if present.
        if "[/INST]" in raw_response:
            answer = raw_response.split("[/INST]", 1)[1].strip()
        else:
            answer = raw_response.strip()
        return answer

# Update this path to your actual FAISS index location.
vector_db_path = "manit_vector_db/faiss_index"
# Model name for Meta-Llama's Llama-3.1-8B-Instruct (4-bit quantized).
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Cache the chatbot so it isn't reloaded on every interaction.
@st.cache_resource(show_spinner=False)
def load_chatbot():
    return LocalMANITChatbot(vector_db_path, model_name)

chatbot = load_chatbot()

st.title("MANIT Knowledge Assistant")
st.markdown("Ask questions about MANIT's departments, research, and publications.")

# Container for the chat conversation.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input.
user_input = st.text_input("Your question:", placeholder="Enter your question here...")

if st.button("Submit") and user_input:
    st.session_state.chat_history.append(("User", user_input))
    
    answer_container = st.empty()  # Container for streaming answer.
    full_answer = chatbot.query(user_input)
    tokens = full_answer.split()  # Tokenize by whitespace.
    cumulative = ""
    for token in tokens:
        cumulative += token + " "
        answer_container.markdown(cumulative)
        time.sleep(0.05)  # Adjust the delay as desired.
    st.session_state.chat_history.append(("Bot", cumulative))
    
# Display the conversation history.
for speaker, text in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"**User:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
