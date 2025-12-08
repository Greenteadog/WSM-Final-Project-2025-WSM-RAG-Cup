import jsonlines
import json
from Connection import Connection
from utils import create_table_from_yaml

SCHEMA_PATH = 'db/dataset_table-schema.yaml'
DB_PATH = 'db/dataset.db'
DATASET_PATH = 'dragonball_dataset/dragonball_docs.jsonl'
SPECIAL_DATASET_PATH = 'db/special_dataset.jsonl'

def create_tables():
    conn = Connection(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS documents")
    create_table_from_yaml(SCHEMA_PATH, DB_PATH)

def main(docs_path):
    create_tables()
    populate_documents(docs_path)
    insert_special_documents()

##################################
# Here for documents table
##################################

def insert_document(doc):
    conn = Connection(DB_PATH)
    domain = doc['domain']
    mapping = {
        "Finance": "company_name",
        "Law": "court_name",
        "Medical": "hospital_patient_name"
    }
    name = doc[mapping[domain]]
    conn.execute(
        "INSERT INTO documents (doc_id, domain, language, name, content, jsonl) VALUES (?, ?, ?, ?, ?, ?)",
        (doc['doc_id'], domain, doc['language'], name, doc['content'], json.dumps(doc))
    )
        

def populate_documents(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        for doc in reader:
            insert_document(doc)

##################################
# Here for handling modified special dataset
##################################

def insert_special_documents():
    with jsonlines.open(SPECIAL_DATASET_PATH, 'r') as reader:
        for doc in reader:
            insert_document(doc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regen', type=bool, default=False, help='Regenerate the database [default: False]')
    parser.add_argument('--docs_path', type=str, default=DATASET_PATH, help='Path to the documents file')
    args = parser.parse_args()
    if (args.regen):
        main(args.docs_path)
    else:
        print("No action taken.")