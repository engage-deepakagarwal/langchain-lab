import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    text_loader = TextLoader(f"{os.getcwd()}/text_loader/medium_blog_post.txt", encoding='utf8')
    document = text_loader.load()

    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="embeddings",
        openai_api_version=os.getenv("OPENAI_API_EMBEDDINGS_VERSION"),
    )
    document_search = PineconeVectorStore.from_documents(texts, embeddings, index_name="medium-blog-post-index")

    llm = AzureChatOpenAI(azure_deployment='turbo', temperature=0.1)
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=document_search.as_retriever(),
        return_source_documents=True)
    query = "What is a vector database? Give me a 15 words answer for a beginner"
    result = retrieval_qa({"query": query})
    print(result)




