import os
import yaml
import hashlib
import pickle
import logging
import pysolr
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Import text extraction utility
from io_utils.pre_processor import (
    extract_text_and_url_from_html,
    clean_text,
    filter_noise
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("solr-sync")

# Solr uses a different state file to track indices
STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "solr_storage", "ingestion_state.pkl")

def get_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file to detect content changes."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_state() -> Dict[str, str]:
    """Load the last known state of files (path -> hash)."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load state file: {e}. Starting fresh.")
    return {}

def save_state(state: Dict[str, str]):
    """Save the current state of files."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(state, f)

def sync_solr():
    """
    Identifies new or changed HTML files and updates the Solr index.
    This script is dedicated exclusively to Solr indexing.
    """
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    data_dir = cfg.get('data', {}).get('data_to_db', 'data_raw')
    solr_url = cfg['retrieval'].get('solr_url', "http://solr:8983/solr/biokdf")
    
    # Ensure paths are relative to workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(workspace_root, data_dir)

    logger.info(f"🚀 Starting Solr Ingestion Sync for {data_dir}")
    logger.info(f"Target Solr: {solr_url}")
    
    # 2. Detect Changes
    current_state = {}
    changed_files = []
    old_state = load_state()

    data_path_obj = Path(data_dir)
    if not data_path_obj.exists():
        logger.error(f"Data directory {data_dir} not found.")
        return

    # Find all HTML files in the raw data directory
    for file_path in data_path_obj.rglob("*.html"):
        str_path = str(file_path)
        file_hash = get_file_hash(str_path)
        current_state[str_path] = file_hash
        
        # Check if file is new or modified
        if str_path not in old_state or old_state[str_path] != file_hash:
            changed_files.append(file_path)

    if not changed_files:
        logger.info("✅ No changes detected for Solr. Index is up to date.")
        return

    logger.info(f"📝 Found {len(changed_files)} new or modified files for Solr.")

    # 3. Initialize Solr Client
    try:
        solr = pysolr.Solr(solr_url, always_commit=True, timeout=10)
    except Exception as e:
        logger.error(f"Failed to connect to Solr: {e}")
        return

    # 4. Process Loop
    for file_path in changed_files:
        str_path = str(file_path)
        relative_path = file_path.relative_to(data_path_obj)
        dataset_id = str(relative_path.with_suffix("")).replace("\\", "/") # Use as ID
        
        logger.info(f"Indexing to Solr: {dataset_id}")
        
        try:
            # Extract content
            raw_html, source_url, source_title = extract_text_and_url_from_html(str_path)
            cleaned = clean_text(raw_html)
            filtered = filter_noise(cleaned)

            if not filtered:
                logger.warning(f"No content extracted from {str_path}")
                continue

            # Populate Solr Document schema
            solr_doc = {
                "id": dataset_id,
                "title": source_title,
                "url": source_url,
                "content": filtered,
                "last_modified": datetime.now().isoformat() + "Z",
                "source_file_path": str_path
            }

            # Add to Solr
            solr.add([solr_doc])
            
        except Exception as e:
            logger.error(f"Failed to index {str_path} to Solr: {e}")
            # If processing fails, remove it from the current state so it gets retried next time
            if str_path in current_state:
                del current_state[str_path]

    # 5. Finalize State
    save_state(current_state)
    logger.info("✨ Solr ingestion sync complete.")

if __name__ == "__main__":
    sync_solr()
