from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings # embeddings (to covert text to vectors)
from langchain.vectorstores import FAISS # facebook semantic (similar) search
from langchain.chains.question_answering import load_qa_chain # question answering chain - https://python.langchain.com/en/latest/use_cases/question_answering.html
from langchain.llms.openai import OpenAI # openai llm
from langchain.callbacks import get_openai_callback # for monitoring token useage only works with openai

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF 🔗📝")
    st.header("Ask your PDF 🔗📝")

# upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

# extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", # end of line
            chunk_size=1000, # characters
            chunk_overlap=200, # more context
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings using function from OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        # Create knowledge base
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # User input
        user_question = st.text_input("Ask a question about your PDF:")
        # Search for relevant chunks
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
        
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb: # for monitoring token useage
                response = chain.run(input_documents=docs, question=user_question) # The Response
                print(cb) # cost/token useage
           
            st.write(response)
        




if __name__ == '__main__':
    main()