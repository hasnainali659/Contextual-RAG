import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
import cohere
import logging
import time
from llama_parse import LlamaParse
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from langchain_community.document_loaders.doc_intelligence import AzureAIDocumentIntelligenceLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv('azure.env', override=True)

class ContextualRetrieval:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        self.embeddings = AzureOpenAIEmbeddings(
                            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                            azure_deployment="text-embedding-ada-002",
                            openai_api_version="2024-03-01-preview",
                            azure_endpoint =os.environ["AZURE_OPENAI_ENDPOINT"]
                        )
        self.llm = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

    def load_pdf_and_parse(self, pdf_path: str) -> str:
        loader = AzureAIDocumentIntelligenceLoader(file_path=pdf_path, 
                                           api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"), 
                                           api_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
                                           api_model="prebuilt-layout",
                                           api_version="2024-02-29-preview",
                                           mode='markdown',
                                           analysis_features = [DocumentAnalysisFeature.OCR_HIGH_RESOLUTION])

        try:
            documents = loader.load()
            if not documents:
                raise ValueError("No content extracted from the PDF.")
            return " ".join([doc.page_content for doc in documents])
        except Exception as e:
            logging.error(f"Error while parsing the file '{pdf_path}': {str(e)}")
            raise

    def process_document(self, document: str) -> Tuple[List[Document], List[Document]]:
        if not document.strip():
            raise ValueError("The document is empty after parsing.")
        chunks = self.text_splitter.create_documents([document])
        contextualized_chunks = self._generate_contextualized_chunks(document, chunks)
        return chunks, contextualized_chunks

    def _generate_contextualized_chunks(self, document: str, chunks: List[Document]) -> List[Document]:
        contextualized_chunks = []
        for chunk in chunks:
            context = self._generate_context(document, chunk.page_content)
            contextualized_content = f"{context}\n\n{chunk.page_content}"
            contextualized_chunks.append(Document(page_content=contextualized_content, metadata=chunk.metadata))
        return contextualized_chunks

    def _generate_context(self, document: str, chunk: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specializing in document analysis. Your task is to provide brief, relevant context for a chunk of text from the given document.
        Here is the document:
        <document>
        {document}
        </document>

        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>

        Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
        1. Identify the main topic or concept discussed in the chunk.
        2. Mention any relevant information or comparisons from the broader document context.
        3. If applicable, note how this information relates to the overall theme or purpose of the document.
        4. Include any key figures, dates, or percentages that provide important context.
        5. Do not use phrases like "This chunk discusses" or "This section provides". Instead, directly state the context.

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.

        Context:
        """)
        messages = prompt.format_messages(document=document, chunk=chunk)
        response = self.llm.invoke(messages)
        return response.content

    def create_bm25_index(self, chunks: List[Document]) -> BM25Okapi:
        tokenized_chunks = [chunk.page_content.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        prompt = ChatPromptTemplate.from_template("""
        Based on the following information, please provide a concise and accurate answer to the question.
        If the information is not sufficient to answer the question, say so.

        Question: {query}

        Relevant information:
        {chunks}

        Answer:
        """)
        messages = prompt.format_messages(query=query, chunks="\n\n".join(relevant_chunks))
        response = self.llm.invoke(messages)
        return response.content

    def rerank_results(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        logging.info(f"Reranking {len(documents)} documents for query: {query}")
        doc_contents = [doc.page_content for doc in documents]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                reranked = self.cohere_client.rerank(
                    model="rerank-english-v2.0",
                    query=query,
                    documents=doc_contents,
                    top_n=top_n
                )
                break
            except cohere.errors.TooManyRequestsError:
                if attempt < max_retries - 1:
                    logging.warning(f"Rate limit hit. Waiting for 60 seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(60)  # Wait for 60 seconds before retrying
                else:
                    logging.error("Rate limit hit. Max retries reached. Returning original documents.")
                    return documents[:top_n]
        
        logging.info(f"Reranking complete. Top {top_n} results:")
        reranked_docs = []
        for idx, result in enumerate(reranked.results):
            original_doc = documents[result.index]
            reranked_docs.append(original_doc)
            logging.info(f"  {idx+1}. Score: {result.relevance_score:.4f}, Index: {result.index}")
        
        return reranked_docs

    def expand_query(self, original_query: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specializing in document analysis. Your task is to expand the given query to include related terms and concepts that might be relevant for a more comprehensive search of the document.

        Original query: {query}

        Please provide an expanded version of this query, including relevant terms, concepts, or related ideas that might help in summarizing the full document. The expanded query should be a single string, not a list.

        Expanded query:
        """)
        messages = prompt.format_messages(query=original_query)
        response = self.llm.invoke(messages)
        return response.content
    
cr = ContextualRetrieval()
pdf_path = "1.pdf"
document = cr.load_pdf_with_llama_parse(pdf_path)

# Process the document
chunks, contextualized_chunks = cr.process_document(document)

# Create BM25 index
contextualized_bm25_index = cr.create_bm25_index(contextualized_chunks)
normal_bm25_index = cr.create_bm25_index(chunks)

def process_query(query: str, processor: AutoProcessor, model: ColPali) -> np.ndarray:
    mock_image = Image.new('RGB', (224, 224), color='white')

    inputs = processor(text=query, images=mock_image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs)

    return torch.mean(embeddings, dim=1).float().cpu().numpy().tolist()[0]

original_query = "When does the term of the Agreement commence and how long does it last?"
print(f"\nOriginal Query: {original_query}")
process_query(cr, original_query, normal_bm25_index, chunks)

original_query = "When does the term of the Agreement commence and how long does it last?"
print(f"\nOriginal Query: {original_query}")
process_query(cr, original_query, contextualized_bm25_index, contextualized_chunks)