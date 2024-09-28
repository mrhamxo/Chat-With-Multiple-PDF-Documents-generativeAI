import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
#from langchain.vectorstores import FAISS 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain.chains.question_answering import load_qa_chain  
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from environment variables
os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""  # Initialize an empty string to hold the extracted text
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Initialize the PDF reader for each PDF file
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page and append it
    return text  # Return the concatenated text

# Function to split long text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,  # Size of each chunk
        chunk_overlap=1000  # Overlap between chunks to maintain context
    )
    chunks = text_splitter.split_text(text)  # Split the text into chunks
    return chunks  # Return the text chunks

# Function to create vector store using FAISS and embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"  # Use the Google Embeddings model
    )
    
    # Create a FAISS vector store from the text chunks and embeddings
    vector_store = FAISS.from_texts(
        text_chunks, 
        embedding=embeddings
    )
    vector_store.save_local("faiss_index")  # Save the vector store locally

# Function to load the conversational chain for answering questions
def get_conversational_chain():

    # Create a prompt template for detailed question answering
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    # Initialize the conversational model from Google Generative AI
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",  # Use the Gemini model for answering questions
        temperature=0.3  # Set model temperature for response diversity
    )

    # Create a prompt with the defined template and variables
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # Load the question answering chain using the conversational model and prompt
    chain = load_qa_chain(
        model, 
        chain_type="stuff",  # Define chain type as "stuff" for straightforward QA
        prompt=prompt
    )

    return chain  # Return the loaded QA chain

# Function to handle user input and generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"  # Use the embeddings model
    )
    
    # Load the FAISS vector store locally using the embeddings
    #new_db = FAISS.load_local("faiss_index", embeddings)
    
    # Load the FAISS vector store locally with dangerous deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Search the most relevant chunks from the vector store based on the user's question
    docs = new_db.similarity_search(user_question)

    # Load the conversational chain for question answering
    chain = get_conversational_chain()

    # Get the response from the QA chain using the relevant documents and the user's question
    response = chain.invoke(
        {"input_documents": docs, "question": user_question}
    )

    # Print the response to the console (for debugging)
    print(response)

    # Display the response in the Streamlit app
    st.write("Reply: ", response["output_text"])

# Main function to run the Streamlit app
def main():
    # Set up the Streamlit page configuration
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")  # App header

    # Input field for the user to ask a question
    user_question = st.text_input("Ask a Question from the PDF Files")

    # If a question is asked, process it
    if user_question:
        user_input(user_question)

    # Sidebar for uploading PDF files and processing them
    with st.sidebar:
        st.title("Menu:")
        
        # Upload multiple PDF files
        pdf_docs = st.file_uploader("Upload PDF Files & Click the Button", accept_multiple_files=True)
        
        # Button to submit and process the PDF files
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)  # Extract text from the uploaded PDFs
                text_chunks = get_text_chunks(raw_text)  # Split the text into chunks
                get_vector_store(text_chunks)  # Create and save the vector store
                st.success("Done")  # Display success message

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
