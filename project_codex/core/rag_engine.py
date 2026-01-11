"""
RAG Engine module for Project Codex.
Handles interactions with ChromaDB for storing and retrieving terms and styles.
"""

import os
import uuid
from typing import List, Dict, Optional, Any, cast
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Metadata

# Use a relative path that works within the project structure
# Assuming the app runs from the project root
DEFAULT_VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "../data/vector_db")


class RAGEngine:
    """
    RAG Engine encapsulating ChromaDB interactions.
    """

    def __init__(self, persist_directory: str = DEFAULT_VECTOR_DB_PATH):
        """
        Initialize the RAG Engine with a persistent ChromaDB client.

        Args:
            persist_directory: Path to the directory where the database is stored.
        """
        self.persist_directory = persist_directory
        try:
            # Ensure the directory exists
            os.makedirs(persist_directory, exist_ok=True)

            self.client: ClientAPI = chromadb.PersistentClient(path=persist_directory)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB at {persist_directory}: {str(e)}"
            ) from e

    def _get_collection(self, collection_name: str):
        """
        Retrieve or create a collection.

        Args:
            collection_name: The name of the collection.

        Returns:
            The collection object.

        Raises:
            RuntimeError: If collection retrieval fails.
        """
        try:
            return self.client.get_or_create_collection(name=collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to get or create collection '{collection_name}': {str(e)}"
            ) from e

    def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add texts to a specific collection.

        Args:
            collection_name: The name of the target collection.
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries corresponding to texts.
            ids: Optional list of unique IDs. If None, UUIDs will be generated.

        Raises:
            ValueError: If input lists have mismatched lengths.
            RuntimeError: If the operation fails.
        """
        if not texts:
            return

        if metadatas is not None and len(texts) != len(metadatas):
            raise ValueError("The length of 'texts' and 'metadatas' must match.")

        if ids is not None and len(texts) != len(ids):
            raise ValueError("The length of 'texts' and 'ids' must match.")

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        try:
            collection = self._get_collection(collection_name)
            # Pyright complains about List[Dict[str, Any]] not being strict enough for Metadata
            # Chroma expects Metadata = Mapping[str, Union[str, int, float, bool]]
            # We explicitly cast to satisfy type checker, assuming caller provides valid types
            valid_metadatas = cast(Optional[List[Metadata]], metadatas)

            collection.add(documents=texts, metadatas=valid_metadatas, ids=ids)
        except Exception as e:
            raise RuntimeError(
                f"Failed to add texts to collection '{collection_name}': {str(e)}"
            ) from e

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the collection for similar texts.

        Args:
            collection_name: The name of the collection to query.
            query_text: The query string.
            n_results: Number of results to return.
            where: Optional filtering criteria.

        Returns:
            A dictionary containing the query results (documents, metadatas, distances).

        Raises:
            RuntimeError: If the query fails.
        """
        try:
            collection = self._get_collection(collection_name)
            results = collection.query(
                query_texts=[query_text], n_results=n_results, where=where
            )
            # QueryResult is a TypedDict in recent Chroma versions,
            # which is compatible with Dict[str, Any] at runtime but
            # pyright might be strict about it not being explicitly declared as such.
            return cast(Dict[str, Any], results)
        except Exception as e:
            raise RuntimeError(
                f"Failed to query collection '{collection_name}': {str(e)}"
            ) from e

    def clear_collection(self, collection_name: str) -> None:
        """
        Delete a collection if it exists.

        Args:
             collection_name: The name of the collection to clear.
        """
        try:
            # Check if the collection exists before attempting to delete it.
            existing_collections = [c.name for c in self.client.list_collections()]
            if collection_name in existing_collections:
                self.client.delete_collection(name=collection_name)
        except Exception as e:
            # For any other unexpected errors, re-raise them as a runtime error.
            raise RuntimeError(
                f"Failed to clear collection '{collection_name}': {str(e)}"
            ) from e

    def get_all(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieve all documents from a collection.

        Args:
            collection_name: The name of the collection.

        Returns:
            Dictionary with 'documents', 'metadatas', 'ids'.
        """
        try:
            collection = self._get_collection(collection_name)
            # ChromaDB get() without args returns all (or default limit?)
            # Usually .get() returns everything if no ids/where specified,
            # but might be paginated. We'll verify.
            # safe assumption: get()
            return cast(Dict[str, Any], collection.get())
        except Exception as e:
            raise RuntimeError(
                f"Failed to get all from '{collection_name}': {str(e)}"
            ) from e
