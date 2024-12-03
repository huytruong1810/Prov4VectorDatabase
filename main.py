import datetime
import os
import json

import pandas as pd
import numpy as np
import psycopg2
import math

import torch
import torch.nn.functional as f
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DOCUMENT_FOLDER_PATH = "./documents/"
TARGET_NUM_TOKENS = 512
EMBED_MODEL_NAME = "all-mpnet-base-v2"
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)  # Does not need to be the same as target LLM
TOKENIZER = EMBED_MODEL.tokenizer
SCHEME = (
    ('id', 'SERIAL PRIMARY KEY'),
    ('filename', 'TEXT'),
    ('chunk_text', 'TEXT'),
    ('start_index', 'INT'),
    ('end_index', 'INT'),
    ('embed_model', 'TEXT'),
    ('num_tokens', 'INT'),
    ('embedding', f'VECTOR({EMBED_MODEL.get_sentence_embedding_dimension()})')
)
PROV_SCHEME = (
    ('chunk_id', 'INT REFERENCES documents (id)'),  # Link to the document table
    ('query_text', 'TEXT'),
    ('embed_model', 'TEXT'),
    ('top_k', 'INT'),
    ('time', 'TIMESTAMP'),
    ('query_influence', 'JSONB')  # Store query token influences
)


def rag_with_provenance(query_text: str, top_k: int, cursor):
    invoke_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query_embedding = np.array(EMBED_MODEL.encode(query_text))
    cursor.execute(f"SELECT id, chunk_text, embedding FROM documents ORDER BY embedding <=> %s LIMIT {top_k}",
                   (query_embedding,))
    results = cursor.fetchall()

    provenance_data = []
    query_tokens = TOKENIZER.tokenize(query_text)
    for chunk_id, chunk_text, chunk_embedding in results:
        # Approximate each token influence
        query_influence = {
            token: f.cosine_similarity(torch.tensor(EMBED_MODEL.encode(token)), torch.tensor(chunk_embedding), dim=0).item()
            for token in query_tokens
        }
        provenance_data.append(
            (chunk_id, query_text, EMBED_MODEL_NAME, top_k, invoke_time, json.dumps(query_influence))
        )

    execute_values(cursor, "INSERT INTO provenance (chunk_id, query_text, embed_model, top_k, time, query_influence) "
                           "VALUES %s;", provenance_data)
    conn.commit()

    return results


def prov_use_case_1(cursor, client_query_pattern):
    query = """
        SELECT p.query_text, d.chunk_text, p.query_influence
        FROM provenance p
        JOIN documents d ON p.chunk_id = d.id
        WHERE p.query_text LIKE %s; 
    """
    cursor.execute(query, (client_query_pattern,))  # Pass the pattern as a parameter
    results = cursor.fetchall()
    print("Use case 1 - see user's queries, retrieved chunks, and their token-level influences")
    for query_text, chunk_text, query_influence in results:
        print(f"Query: {query_text}")
        print(f"Chunk: {chunk_text}")
        print(f"Influences: {json.dumps(query_influence, indent=4)}")  # Parse JSONB


def prov_use_case_2(cursor):
    query = """
        SELECT d.filename, d.start_index, d.end_index, d.chunk_text
        FROM provenance p
        JOIN documents d ON p.chunk_id = d.id
        WHERE p.time = (SELECT MAX(time) FROM provenance); -- Latest query
    """
    cursor.execute(query)
    results = cursor.fetchall()
    print("Use case 2 - see where in the original document the retrieved chunks for the latest query are")
    for filename, start_index, end_index, chunk_text in results:
        print(f"File: {filename}, Start: {start_index}, End: {end_index}, Text: {chunk_text}")


def get_embedding_data(directory) -> list:
    data = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            with open(DOCUMENT_FOLDER_PATH + filename, 'r', encoding="utf8") as file_content:
                text = file_content.read()
                max_seq_length = EMBED_MODEL.get_max_seq_length()
                all_tokens = []  # Initialize the holder of all tokens
                for i in range(0, len(text), max_seq_length):  # Iterate by character length
                    sub_text = text[i: min(i + max_seq_length, len(text))]
                    sub_tokens = TOKENIZER.encode(sub_text, add_special_tokens=False)  # tokenize sub_text
                    all_tokens.extend(sub_tokens)  # extend to include all tokens in document
                num_chunks = math.ceil(len(all_tokens) / TARGET_NUM_TOKENS)  # Token-wise chunking
                for i in range(num_chunks):  # Now we create embeddings
                    start_index = i * TARGET_NUM_TOKENS
                    end_index = min((i + 1) * TARGET_NUM_TOKENS, len(all_tokens))
                    chunk_tokens = all_tokens[start_index:end_index]
                    chunk_text = TOKENIZER.decode(chunk_tokens)
                    data.append([filename, chunk_text, start_index, end_index, EMBED_MODEL_NAME,
                                 len(chunk_tokens), EMBED_MODEL.encode(chunk_text).tolist()])
    return data


if __name__ == "__main__":
    # Create Pandas DataFrame
    column_names = [col[0] for col in SCHEME]
    df = pd.DataFrame(
        data=get_embedding_data(os.fsencode(DOCUMENT_FOLDER_PATH)),
        columns=list(column_names[1:])  # Exclude the id column
    )
    # Print to check dataframe
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    conn = cursor = None
    try:  # Use try/except block for robust error handling
        conn = psycopg2.connect(database="postgres",
                                user="postgres",
                                host='localhost',
                                password="your password",
                                port=5432)
        cursor = conn.cursor()
        # Install pgvector
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

        # Register the pgvector extension
        register_vector(conn)

        # Create the table
        cursor.execute(
            query=f"DROP TABLE IF EXISTS documents CASCADE;"
        )
        conn.commit()
        cursor.execute(
            query=f"CREATE TABLE IF NOT EXISTS documents ({', '.join(f'{col[0]} {col[1]}' for col in SCHEME)});"
        )
        conn.commit()

        # Create provenance table
        cursor.execute(
            query=f"DROP TABLE IF EXISTS provenance CASCADE;"
        )
        cursor.execute(
            query=f"CREATE TABLE IF NOT EXISTS provenance ({', '.join(f'{col[0]} {col[1]}' for col in PROV_SCHEME)});"
        )
        conn.commit()

        # Insert data
        execute_values(cursor,
                       "INSERT INTO documents (filename, chunk_text, start_index, end_index, embed_model, num_tokens, "
                       "embedding) VALUES %s;",
                       list(df.itertuples(index=False)))
        conn.commit()

        # Sanity check
        cursor.execute("SELECT COUNT(*) FROM documents;")
        num_records = cursor.fetchone()[0]
        print("Number of records in table: ", num_records, "\n")

        # Test retrieval
        print(rag_with_provenance(
            query_text=input("Enter a query for the LLM: "),
            top_k=int(input("Enter the number k in top_k chunks to retrieve:")),
            cursor=cursor)
        )

        prov_use_case_1(cursor, client_query_pattern='%blues%')
        prov_use_case_2(cursor)

    except psycopg2.Error as e:
        print(f"PostgreSQL Error: {e}")
    finally:  # Ensure connection is closed even if errors occur
        if cursor:
            cursor.close()
        if conn:
            conn.close()
