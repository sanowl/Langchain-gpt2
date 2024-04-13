from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from langchain.llms import HuggingFacePipeline
# Assuming the following modules exist in the version of langchain you're using
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.preprocessors import BasicPreprocessor

# Load the GPT-2 model and tokenizer once to avoid repeated loading
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Create a Hugging Face pipeline for text generation
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Create a Hugging Face LLM wrapper
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Function to preprocess text
def preprocess_text(text):
    preprocessor = BasicPreprocessor()
    return preprocessor.preprocess(text)

# Function to split text and create a vector store
# This function will need to be updated to use a different text splitting approach
# if CharacterTextSplitter is not available in langchain
def prepare_text_vector_store(text):
    # Placeholder for text splitting logic
    texts = [text]  # Simple example, split your text as needed
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(texts, embeddings)

# Main function to handle user input and process the QA chain
def main():
    text = input("Enter the text you want to ask questions about: ")
    preprocessed_text = preprocess_text(text)
    try:
        docsearch = prepare_text_vector_store(preprocessed_text)
    except Exception as e:
        print(f"An error occurred while preparing the text vector store: {e}")
        return

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        input_key="question"
    )

    while True:
        question = input("Ask a question about the text (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        try:
            answer = qa_chain({"question": question})
            print("Answer:", answer['result'])
        except Exception as e:
            print(f"An error occurred while answering the question: {e}")

if __name__ == "__main__":
    main()