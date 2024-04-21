import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    pdf_path = f"{os.getcwd()}/pdf_loader/2210.03629.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=3)
    docs = text_splitter.split_documents(document)

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="embeddings",
        openai_api_version=os.getenv("OPENAI_API_EMBEDDINGS_VERSION"),
    )

    document_search = FAISS.from_documents(docs, embeddings)

    # You can optionally save the vector store index locally on the hard-disk and load from it.
    # Else by default, it gets saved in RAM
    # document_search.save_local("faiss_index_react")
    # document_search = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    llm = AzureChatOpenAI(azure_deployment='turbo', temperature=0.1)
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=document_search.as_retriever(),
        return_source_documents=True)

    query = "Give me a gist of ReAct in 3 sentences"
    result = retrieval_qa({"query": query})
    print(result)