from typing import List, Dict, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from .embeddings import LocalEmbeddings

class VectorStore:
    """Manages the vector database for document storage and retrieval"""
    
    def __init__(self, persist_directory: str = "./data/vectorstore"):
        self.persist_directory = persist_directory
        self.collection_name = "financial_documents"
        
        # Initialize embeddings
        self.embeddings = LocalEmbeddings()
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "Financial documents and reports",
                "hnsw:space": "cosine"  # Use cosine similarity
            }
        )
    
    def add_documents(self, 
                     chunks: List[Dict],
                     source: str,
                     document_type: str = "report") -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of processed chunks with content
            source: Source of the document (e.g., company name)
            document_type: Type of document (e.g., "annual_report", "quarterly_report")
        """
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = f"{source}_{document_type}_{datetime.now().strftime('%Y%m%d')}_{i}"
            
            # Prepare data for insertion
            documents.append(chunk["content"])
            metadatas.append({
                "source": source,
                "document_type": document_type,
                "chunk_index": i,
                **chunk.get("metadata", {})
            })
            ids.append(chunk_id)
            
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(documents)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self,
              query: str,
              n_results: int = 5,
              source: Optional[str] = None,
              document_type: Optional[str] = None) -> List[Dict]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            source: Filter by source (e.g., company name)
            document_type: Filter by document type
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Prepare where clause for filtering
        where = {}
        if source:
            where["source"] = source
        if document_type:
            where["document_type"] = document_type
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where or None
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "id": results["ids"][0][i]
            })
        
        return formatted_results
    
    def delete_documents(self, 
                        source: Optional[str] = None,
                        document_type: Optional[str] = None) -> None:
        """
        Delete documents from the vector store
        
        Args:
            source: Delete documents from this source
            document_type: Delete documents of this type
        """
        where = {}
        if source:
            where["source"] = source
        if document_type:
            where["document_type"] = document_type
        
        if where:
            self.collection.delete(where=where)
        
    def get_similar_chunks(self, chunk_id: str, n_results: int = 5) -> List[Dict]:
        """
        Get similar chunks to a given chunk
        
        Args:
            chunk_id: ID of the chunk to find similar chunks for
            n_results: Number of similar chunks to return
        """
        # Get the original chunk
        results = self.collection.get(ids=[chunk_id])
        if not results["documents"]:
            return []
        
        # Generate embedding for the chunk
        chunk_embedding = self.embeddings.embed_query(results["documents"][0])
        
        # Search for similar chunks using the embedding
        similar = self.collection.query(
            query_embeddings=[chunk_embedding],
            n_results=n_results + 1  # Add 1 because the original chunk will be included
        )
        
        # Format and filter out the original chunk
        formatted_results = []
        for i in range(len(similar["documents"][0])):
            if similar["ids"][0][i] != chunk_id:
                formatted_results.append({
                    "content": similar["documents"][0][i],
                    "metadata": similar["metadatas"][0][i],
                    "distance": similar["distances"][0][i],
                    "id": similar["ids"][0][i]
                })
        
        return formatted_results[:n_results]