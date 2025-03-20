"""
############################# DocuRAG #############################
---------------------------- Clean Database Script ---------------------------

Description:
This script safely removes the document embeddings collection stored in ChromaDB.
It is useful when you need to perform a complete cleanup of the vector database
before indexing new documents, ensuring that outdated data does not interfere with
search and context retrieval. Additionally, it automatically deletes the log file
'processed_files.txt'.

Functionality:
1. Configuration Loading:
   - Loads configuration from the 'config.yml' file, defining BASE_PATH and the
     ChromaDB database path.
   - Constructs the full database path for easier access.

2. Deletion Confirmation:
   - Prompts the user to confirm deletion of the database. Only proceeds if the user
     confirms (by entering "s").

3. Database Cleanup:
   - Connects to ChromaDB using the specified path and deletes the collection named
     "document_embeddings".
   - Displays an error message if the deletion fails.

4. Log File Removal:
   - Deletes the 'processed_files.txt' log file, if it exists.

Expected Output:
- Success message: "Document embeddings collection deleted."
- Cancellation message: "Operation cancelled. The database and 'processed_files.txt' were not deleted."

Notes:
- Use this script only when a complete database cleanup is required.
- Ensure that you have backed up any relevant data before performing this irreversible process.
- The 'config.yml' file must contain a valid 'db_path' under the 'document_vectorizer' section.
"""

import warnings
warnings.filterwarnings("ignore")
import chromadb
import yaml
import os

def load_config(config_path="config.yml"):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    base_path = config.get("BASE_PATH", ".")
    db_path = os.path.join(base_path, config["document_vectorizer"]["db_path"])
    return base_path, db_path

def clear_chroma_collection(db_path):
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(name="document_embeddings")
        print("Document embeddings collection deleted.")
    except Exception as e:
        print("Error deleting the collection:", str(e))

def delete_processed_files_log(base_path):
    log_path = os.path.join(base_path, "processed_files.txt")
    if os.path.exists(log_path):
        try:
            os.remove(log_path)
            print("Log file 'processed_files.txt' deleted.")
        except Exception as e:
            print("Error deleting 'processed_files.txt':", str(e))
    else:
        print("The log file 'processed_files.txt' does not exist or has already been deleted.")

if __name__ == "__main__":
    base_path, db_path = load_config()
    respuesta = input("Do you want to delete the embeddings database in ChromaDB and the 'processed_files.txt' file? (s/n): ").strip().lower()
    if respuesta == "s":
        clear_chroma_collection(db_path)
        delete_processed_files_log(base_path)
    else:
        print("Operation cancelled. The database and 'processed_files.txt' were not deleted.")
