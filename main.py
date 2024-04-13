from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedQAAlgorithm:
    def __init__(self, model_name="gpt2", embedding_model="HuggingFace", chain_type="stuff"):
        # Initialize the class with the specified model, embedding model, and chain type
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chain_type = chain_type
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.hf_pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.llm = HuggingFacePipeline(pipeline=self.hf_pipeline)

    def load_text_from_file(self, file_path):
        # Load text from a file
        with open(file_path, 'r') as file:
            return file.read()

    def prepare_text_vector_store(self, text):
        # Split the text into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        
        # Choose the appropriate embedding model based on the specified embedding_model
        if self.embedding_model == "OpenAI":
            embeddings = OpenAIEmbeddings()
        elif self.embedding_model == "SentenceTransformer":
            embeddings = SentenceTransformerEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings()
        
        # Create a vector store using FAISS
        return FAISS.from_texts(texts, embeddings)

    def process_qa_chain(self, docsearch, question):
        # Create a RetrievalQA chain using the specified chain type
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.chain_type,
            retriever=docsearch.as_retriever(),
            input_key="question"
        )
        # Process the question and return the answer
        return qa_chain({"question": question})['result']

    def save_answers_to_file(self, answers, output_file):
        # Save the answers to a file
        with open(output_file, 'w') as file:
            for answer in answers:
                file.write(answer + '\n')

def main():
    # Create an instance of the AdvancedQAAlgorithm with the desired model, embedding model, and chain type
    algo = AdvancedQAAlgorithm(model_name="gpt2-medium", embedding_model="SentenceTransformer", chain_type="map_reduce")

    # Get the file path from user input
    file_path = input("Enter the path to the text file: ")
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return

    # Load the text from the file
    text = algo.load_text_from_file(file_path)
    
    # Prepare the text vector store
    try:
        docsearch = algo.prepare_text_vector_store(text)
    except Exception as e:
        logger.error(f"An error occurred while preparing the text vector store: {e}")
        return

    # Collect questions from the user
    questions = []
    while True:
        question = input("Ask a question about the text (or press Enter to finish): ")
        if question == "":
            break
        questions.append(question)

    # Process the questions and store the answers
    answers = []
    for question in questions:
        try:
            answer = algo.process_qa_chain(docsearch, question)
            answers.append(answer)
            print(f"Question: {question}")
            print(f"Answer: {answer}\n")
        except Exception as e:
            logger.error(f"An error occurred while answering the question: {e}")

    # Save the answers to a file
    output_file = 'answers.txt'
    algo.save_answers_to_file(answers, output_file)
    print(f"Answers saved to {output_file}")

if __name__ == "__main__":
    main()