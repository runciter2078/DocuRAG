# -*- coding: utf-8 -*-
"""
############################# DocuRAG #############################
------------------------ Document QA Script ------------------------

Description:
This script allows users to ask questions (in Spanish) about documents stored in a vector database (ChromaDB).
It processes documents (which may be in English or Spanish) to provide accurate answers.

Functionality:
1. Querying ChromaDB:
   - The user inputs a question in Spanish.
   - The question is converted into an embedding using a multilingual SentenceTransformer ('LaBSE').
   - The ChromaDB database is queried to retrieve the most relevant contexts based on embedding similarity.

2. Context Processing:
   - Retrieved contexts are cleaned to remove extra spaces and special characters.
   - If the context is in English, it is automatically translated to Spanish using a translation pipeline.

3. Answer Generation:
   - A Spanish QA model ('mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es') generates an answer based on the combined context.
   - The context is split into manageable fragments to fit the model's maximum input length.
   - The best answer is selected based on the model's score.

4. User Output:
   - The generated answer and the relevant document contexts are displayed to the user.

Expected Output:
- A generated answer based on the stored documents.
- Display of the relevant context segments used for generating the answer.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import chromadb
import spacy
import re
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import yaml

def load_config(config_file='config.yml'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_file)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

qa_config = config.get('question_answering', {})
BASE_PATH = config.get('BASE_PATH', '.')
DB_PATH = os.path.join(BASE_PATH, qa_config.get('db_path', 'sqlite_folder'))
MODEL_PATH = os.path.join(BASE_PATH, qa_config.get('qa_model_path', 'qa_model'))
MODEL_NAME = qa_config.get('qa_model_name', 'mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es')

SPACY_MODEL = qa_config.get('spacy_model', 'es_core_news_sm')
TRANSLATION_MODEL = qa_config.get('translation_model', 'Helsinki-NLP/opus-mt-en-es')
SENTENCE_TRANSFORMER_MODEL_QA = qa_config.get('sentence_transformer_model_qa', 'sentence-transformers/LaBSE')

QA_N_RESULTS = qa_config.get('qa_n_results', 3)
QA_CONTEXT_LENGTH = qa_config.get('qa_context_length', 1500)
QA_MAX_ANSWER_LENGTH = qa_config.get('qa_max_answer_length', 200)
QA_MAX_CONTEXT_TOKENS = qa_config.get('qa_max_context_tokens', 450)

EXCLUDE_SOURCES = config.get('common', {}).get('exclude_sources', [])

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_folder(DB_PATH)
create_folder(MODEL_PATH)

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    print(f"The spaCy model '{SPACY_MODEL}' is not installed. Run 'python -m spacy download {SPACY_MODEL}'")
    exit(1)

if 'language_detector' not in Language.factories:
    @Language.factory('language_detector')
    def create_language_detector(nlp, name):
        return LanguageDetector()
    
if 'language_detector' not in nlp.pipe_names:
    nlp.add_pipe('language_detector', last=True)

try:
    translator_en_es = pipeline("translation", model=TRANSLATION_MODEL)
except Exception as e:
    print(f"Error loading the translation model '{TRANSLATION_MODEL}': {e}")
    exit(1)

sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_QA)

def translate_text(text, translator, max_length=512):
    max_chunk_length = 512
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    translated_text = ''
    for chunk in chunks:
        try:
            translated = translator(chunk, max_length=max_length)
            translated_text += translated[0]['translation_text'] + ' '
        except Exception as e:
            print(f"Translation error: {e}")
    return translated_text.strip()

def clean_context(text):
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\sáéíóúÁÉÍÓÚñÑüÜ.,;:()\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model():
    config_file = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_file):
        print(f"Model not complete in {MODEL_PATH}. Downloading from Hugging Face...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
            tokenizer.save_pretrained(MODEL_PATH)
            model.save_pretrained(MODEL_PATH)
            print(f"Model downloaded and saved in {MODEL_PATH}.")
        except Exception as e:
            print(f"Error downloading the model: {e}")
            exit(1)
    else:
        print(f"Model already exists in {MODEL_PATH}. Loading from local...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
        except Exception as e:
            print(f"Error loading the model from {MODEL_PATH}: {e}")
            exit(1)
    return model, tokenizer

def detect_language(text):
    doc = nlp(text)
    language = doc._.language['language']
    return language

def split_context(context, max_tokens, tokenizer):
    sentences = re.split(r'(?<=[.!?])\s+', context)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        if len(tokenizer.tokenize(current_chunk)) + len(sentence_tokens) <= max_tokens:
            current_chunk += ' ' + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_answer(context, question, tokenizer, model):
    max_input_length = tokenizer.model_max_length
    question_tokens = len(tokenizer.tokenize(question))
    max_context_length = max_input_length - question_tokens - 3

    context_chunks = split_context(context, max_context_length, tokenizer)
    qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)
    answers = []
    for chunk in context_chunks:
        inputs = {'question': question, 'context': chunk}
        try:
            result = qa_pipeline(inputs, max_answer_len=QA_MAX_ANSWER_LENGTH)
            answer_text = result['answer'].strip()
            score = result['score']
            answers.append((answer_text, score))
        except Exception as e:
            print(f"Error generating answer: {e}")

    unique_answers = list({ans[0]: ans for ans in answers if ans[0]}.values())

    if not unique_answers:
        return "No answer found in the provided context."

    best_answer, best_score = max(unique_answers, key=lambda x: x[1])
    return best_answer

def query_chroma_collection(collection, query_text, model, n_results=QA_N_RESULTS, context_length=QA_CONTEXT_LENGTH, exclude_sources=None):
    query_embedding = model.encode([query_text])[0]
    where_filter = {}
    if exclude_sources:
        where_filter = {"source": {"$nin": exclude_sources}}
    
    try:
        term_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances'],
            where=where_filter
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return "", [], []

    if not term_results['documents']:
        return "", [], []

    results = list(zip(
        term_results['documents'][0],
        term_results['metadatas'][0],
        term_results['distances'][0]
    ))

    sorted_results = sorted(results, key=lambda x: x[2])
    sources, contexts = set(), []
    total_length = 0

    for doc_text, metadata, distance in sorted_results:
        sources.add(metadata['source'])
        clean_text_segment = clean_context(doc_text)
        if total_length + len(clean_text_segment) <= context_length:
            contexts.append(clean_text_segment)
            total_length += len(clean_text_segme
