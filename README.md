# DocuRAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A modular pipeline for document embedding, retrieval, and question answering using ChromaDB and state-of-the-art transformer models.**

---

## Overview

DocuRAG is a demonstration project that implements a Retrieval-Augmented Generation (RAG) pipeline. It covers the entire process from converting documents to embeddings, storing them in a vector database, and finally performing question answering (QA) on the stored data.

This project leverages several powerful tools and libraries:
- **ChromaDB** for managing vector embeddings.
- **SentenceTransformers (LaBSE)** for generating embeddings.
- **Transformers** for question answering and translation.
- **spaCy** for natural language processing and language detection.
- Other libraries such as **PyMuPDF**, **python-docx**, **python-pptx**, and **nltk** for document processing.

---

## Features

- **Document Processing:** Monitors a folder for new documents (PDF, DOCX, TXT, PPTX), converts them to text, and splits the text into chunks.
- **Embedding Generation:** Uses the LaBSE model to generate semantic embeddings from document text.
- **Vector Storage:** Stores document embeddings in a ChromaDB collection to support efficient similarity searches.
- **Question Answering:** Processes user queries in Spanish by retrieving relevant document contexts and generating answers using a QA model.
- **Logging:** Keeps track of processed files to avoid duplicate processing.

---

## Repository Structure

```
DocuRAG/
├── clean_database.py         # Script to clean the embeddings collection and log file
├── document_vectorizer.py    # Script to process documents and store embeddings in ChromaDB
├── document_qa.py            # Script to perform question answering on stored document embeddings
├── config.yml                # Configuration file for paths and model parameters
└── README.md                 # This file
```

---

## Requirements

- **Python:** 3.8 or above (recommended Python 3.11)
- **Libraries:**
  - chromadb
  - PyYAML
  - PyMuPDF
  - sentence-transformers
  - python-docx
  - python-pptx
  - nltk
  - spacy
  - transformers
  - spacy-langdetect
  - (and their dependencies)

You can install all required packages using `pip` or by creating a virtual environment with a provided `requirements.txt` (if added).

---

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/runciter2078/DocuRAG.git
   cd DocuRAG
   ```

2. **(Optional) Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   If you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Otherwise, install the libraries manually:
   ```bash
   pip install chromadb PyYAML PyMuPDF sentence-transformers python-docx python-pptx nltk spacy transformers spacy-langdetect
   ```

4. **Download Necessary Data:**

   - For **nltk**, ensure the `punkt` tokenizer is downloaded. This will happen automatically on first run.
   - For **spaCy**, download the Spanish model if not already installed:
     ```bash
     python -m spacy download es_core_news_sm
     ```

5. **Configuration:**

   Edit the `config.yml` file if needed. The default settings use relative paths and generic parameters suitable for testing and demonstration.

---

## Usage

### 1. Cleaning the Database

Before processing new documents, you may want to clean the current embeddings collection and log file.

Run:
```bash
python clean_database.py
```
Follow the on-screen prompt to confirm the deletion.

### 2. Document Vectorization

Place your documents (PDF, DOCX, TXT, PPTX) in the folder specified by `data_path` (default is `Data`).

Then, run:
```bash
python document_vectorizer.py
```
This script will process new or modified files, generate embeddings, and store them in ChromaDB.

### 3. Question Answering

Once the documents are processed and embeddings are stored, you can ask questions (in Spanish) based on the stored data.

Run:
```bash
python document_qa.py
```
Enter your question when prompted. The script will retrieve relevant contexts and generate an answer.

---

## Additional Notes

- **Safety:** Use the cleaning script with caution as it will permanently remove the stored embeddings and log file.
- **Customization:** Feel free to modify `config.yml` to suit your directory structure and model preferences.
- **Extensibility:** DocuRAG is modular; you can integrate additional processing steps or swap models as needed.

---

## License

MIT License [https://opensource.org/license/mit]

---

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

---

Enjoy exploring DocuRAG and showcasing your work!
```
