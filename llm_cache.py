import hashlib
from typing import Optional
import sqlite3
import json
import pandas as pd

from typing import Dict, List, Any



def load_cache(db_path: str):
    """
    Load the cache database into a pandas DataFrame for analysis.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        DataFrame containing the cache data
    """
    conn = sqlite3.connect(db_path)

    # Query the data
    query = """
            SELECT request_hash, \
                   model, \
                   prompt_prompt, \
                   messages, \
                   response, timestamp
            FROM llm_cache \
            """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return


class LLMCache:
    """
    A cache for LLM requests using SQLite to avoid redundant API calls.

    This cache stores both the input prompts and output responses from LLM API calls,
    allowing for reproducible experiments and cost savings.
    """

    def __init__(self, db_path: str = "llm_cache.db"):
        """
        Initialize the LLM cache.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the database table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create table to store requests and responses
        cursor.execute(
            """
                       CREATE TABLE IF NOT EXISTS llm_cache
                       (
                           request_hash
                           TEXT
                           PRIMARY
                           KEY,
                           model
                           TEXT,
                           prompt_prompt
                           TEXT,
                           messages
                           TEXT,
                           response
                           TEXT,
                           timestamp
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       """
        )

        conn.commit()
        conn.close()

    def _compute_hash(
        self, model: str, prompt: str, messages: List[Dict[str, str]]
    ) -> str:
        """
        Compute a unique hash for a request based on model, prompt, and messages.

        Args:
            model: The LLM model name
            prompt: The prompt prompt

        Returns:
            A unique hash string for this request
        """


        # Create a hash of the combined inputs
        combined = f"{model}|{prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(
        self, model: str, prompt: str, messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if it exists.

        Args:
            model: The LLM model name
            prompt: The prompt
            messages: The list of message dictionaries

        Returns:
            The cached response or None if not found
        """
        request_hash = self._compute_hash(model, prompt, messages)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT response FROM llm_cache WHERE request_hash = ?", (request_hash,)
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return json.loads(result[0])

        return None

    def store(
        self,
        model: str,
        prompt: str,
        messages: List[Dict[str, str]],
        response: Dict[str, Any],
    ):
        """
        Store a request-response pair in the cache.

        Args:
            model: The LLM model name
            prompt: The prompt
            messages: The list of message dictionaries
            response: The response from the LLM
        """
        request_hash = self._compute_hash(model, prompt, messages)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR REPLACE INTO llm_cache (request_hash, model, prompt_prompt, messages, response) VALUES (?, ?, ?, ?, ?)",
            (request_hash, model, prompt, json.dumps(messages), json.dumps(response)),
        )

        conn.commit()
        conn.close()

    def clear(self):
        """Clear all cached entries from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM llm_cache")

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM llm_cache")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT model, COUNT(*) FROM llm_cache GROUP BY model")
        model_counts = {model: count for model, count in cursor.fetchall()}

        conn.close()

        return {"total_entries": total, "by_model": model_counts}
