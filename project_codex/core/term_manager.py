"""
Term Manager for Project Codex.
Handles loading and saving of glossary terms and synchronization with ChromaDB.
"""

import csv
import json
import os
import uuid
from typing import List, Dict, Any, Optional
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

    def export_terms_to_csv(
        self, file_path: str, collection_name: str = "glossary"
    ) -> None:
        """
        Export terms from RAG collection to CSV.

        Args:
            file_path: Output CSV path.
            collection_name: RAG collection name.
        """
        try:
            data = self.rag_engine.get_all(collection_name)
            metadatas = data.get("metadatas", [])

            # If metadatas is None or empty, we can't reconstruct fully if we relied on it.
            # But add_texts stores 'metadatas'.
            if not metadatas:
                # Fallback: maybe parse documents? But we preferred structured metadata.
                # If list is empty, write headers only.
                self.save_terms_to_csv(file_path, [])
                return

            terms = []
            for meta in metadatas:
                # meta is a Dict[str, Any]
                if meta:
                    terms.append(
                        {
                            "original": str(meta.get("original", "")),
                            "translation": str(meta.get("translation", "")),
                            "remark": str(meta.get("remark", "")),
                        }
                    )

            self.save_terms_to_csv(file_path, terms)

        except Exception as e:
            raise RuntimeError(f"Failed to export terms: {e}") from e


class CharacterManager:
    """
    Manages character sheets, saving to JSON and syncing with RAG.
    """

    def __init__(self, rag_engine: RAGEngine, data_dir: Optional[str] = None):
        """
        Initialize CharacterManager.
        """
        if data_dir is None:
             data_dir = os.path.join(os.path.dirname(__file__), "../data/characters")
        self.rag_engine = rag_engine
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def save_character(self, char_data: Dict[str, Any]) -> None:
        """Saves character to JSON and RAG."""
        name = char_data.get("name")
        if not name:
            raise ValueError("Character name is required.")

        # Save JSON
        file_path = os.path.join(self.data_dir, f"{name}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(char_data, f, ensure_ascii=False, indent=2)

        # Sync to RAG
        text_rep = (
            f"Name: {name}\n"
            f"Bio: {char_data.get('bio', '')}\n"
            f"Tags: {char_data.get('tags', '')}\n"
            f"Quotes: {char_data.get('quotes', '')}"
        )

        # Flatten metadata for Chroma (values must be str, int, float, bool)
        metadata = {
            "name": name,
            "tags": char_data.get("tags", ""),
            "type": "character"
        }

        self.rag_engine.upsert_texts(
            collection_name="characters",
            texts=[text_rep],
            metadatas=[metadata],
            ids=[name]
        )

    def list_characters(self) -> List[str]:
        """List available character names."""
        if not os.path.exists(self.data_dir):
            return []
        return [f.replace(".json", "") for f in os.listdir(self.data_dir) if f.endswith(".json")]

    def load_character(self, name: str) -> Dict[str, Any]:
        """Load character data."""
        path = os.path.join(self.data_dir, f"{name}.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
