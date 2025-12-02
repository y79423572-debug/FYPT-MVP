"""
Term Manager for Project Codex.
Handles loading and saving of glossary terms and synchronization with ChromaDB.
"""
import csv
import os
import uuid
from typing import List, Dict
from project_codex.core.rag_engine import RAGEngine

TERM_COLUMNS = ["original", "translation", "remark"]


class TermManager:
    """
    Manages glossary terms, including CSV I/O and synchronization with RAG.
    """

    def __init__(self, rag_engine: RAGEngine):
        """
        Initialize TermManager.

        Args:
            rag_engine: Instance of RAGEngine for vector DB operations.
        """
        self.rag_engine = rag_engine

    def load_terms_from_csv(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load terms from a CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            List of term dictionaries.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If CSV format is invalid.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Term file not found: {file_path}")

        terms: List[Dict[str, str]] = []
        try:
            with open(file_path, "r", encoding="utf-8", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                # Validate headers
                if not reader.fieldnames:
                    raise ValueError("CSV file is empty or missing headers.")

                # Check if required columns exist
                missing_cols = [
                    col for col in TERM_COLUMNS if col not in reader.fieldnames
                ]
                if missing_cols:
                    raise ValueError(f"Missing columns in CSV: {missing_cols}")

                for row in reader:
                    # Basic validation of row content
                    if not row.get("original") or not row.get("translation"):
                        continue
                    # Skip empty rows or incomplete terms? Or raise?
                    # "Fail Fast" implies raising on bad data if strict,
                    # but skipping empty lines is common.
                    # Let's check if 'original' is empty.

                    term_data = {
                        "original": row["original"].strip(),
                        "translation": row["translation"].strip(),
                        "remark": row.get("remark", "").strip(),
                    }
                    if not term_data["original"]:
                        continue

                    terms.append(term_data)

        except csv.Error as e:
            raise ValueError(f"Failed to parse CSV file: {e}") from e
        except ValueError:
            # Re-raise ValueError directly without wrapping
            raise
        except Exception as e:
            raise RuntimeError(f"Error loading terms: {e}") from e

        return terms

    def save_terms_to_csv(self, file_path: str, terms: List[Dict[str, str]]) -> None:
        """
        Save terms to a CSV file.

        Args:
            file_path: Path to save the CSV.
            terms: List of term dictionaries.
        """
        try:
            with open(file_path, "w", encoding="utf-8", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=TERM_COLUMNS)
                writer.writeheader()
                for term in terms:
                    writer.writerow(term)
        except Exception as e:
            raise RuntimeError(f"Failed to save terms to {file_path}: {e}") from e

    def sync_terms_to_rag(
        self, file_path: str, collection_name: str = "glossary"
    ) -> None:
        """
        Sync terms from CSV to RAG (Glossary Collection).
        This involves clearing the collection and re-adding all terms.

        Args:
            file_path: Path to the CSV file.
            collection_name: Name of the RAG collection.
        """
        terms = self.load_terms_from_csv(file_path)

        # Prepare data for RAG
        texts = []
        metadatas = []
        ids = []

        for term in terms:
            # Format: "Original: {original}\nTranslation: {translation}\nRemark: {remark}"
            # This text representation allows the LLM to retrieve the full context.
            text_rep = (
                f"Original: {term['original']}\n"
                f"Translation: {term['translation']}\n"
                f"Remark: {term['remark']}"
            )
            texts.append(text_rep)
            metadatas.append(term)
            ids.append(term["original"])
            # Use original term as ID to prevent duplicates if we were upserting,
            # but here we are clearing/reloading.
            # NOTE: Chroma IDs must be unique. If a term appears twice in CSV,
            # this might fail. We should handle duplicates or just let it fail/overwrite?
            # Using UUID is safer if duplicates exist, but ID=term is better for lookup.
            # Let's stick to unique check or uuid.
            # Actually, if we use UUID, we can support multiple definitions.
            # But usually glossary implies unique keys.
            # Let's use string of original term for now, assuming uniqueness.
            # If duplicates in CSV, we might want to dedupe or error.

        # Deduplication check
        if len(set(ids)) != len(ids):
            # Fallback to UUIDs or handle duplicates?
            # Let's just generate UUIDs to be safe and avoid crashing on duplicates,
            # though logically a glossary shouldn't have dupe keys.
            ids = [str(uuid.uuid4()) for _ in texts]

        # 1. Clear Collection
        self.rag_engine.clear_collection(collection_name)

        # 2. Add Texts
        if texts:
            self.rag_engine.add_texts(
                collection_name=collection_name,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )
