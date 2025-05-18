import json
import argparse
from sentence_transformers import SentenceTransformer
from utils import prepare_document

def encode(docs, model_name='all-MiniLM-L6-v2', device='cpu'):
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(docs)
    return embeddings

def prepare_vectors(paths, model_name='all-MiniLM-L6-v2', device='cpu', verbose=False):
    docs = []

    for path in paths:
        if verbose:
            print(f"Loading: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for q, a in data.items():
            doc = prepare_document(q, a)
            docs.append(doc)

    if verbose:
        print(f"Encoding {len(docs)} documents...")

    embeddings = encode(docs, model_name=model_name, device=device)
    vector_store = {
		'docs': docs,
		'vectors': embeddings.tolist()
	}

    return vector_store

"""
python prepare_vectors.py --paths processed_data/excel_output.json processed_data/json_output.json --output processed_data/vector_store.json --verbose
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', required=True, help="List of input JSON file paths")
    parser.add_argument('--output', type=str, required=True, help="Path to output JSON file")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('--device', type=str, default='cpu', help="Device for SentenceTransformer (cpu or cuda)")
    args = parser.parse_args()

    vector_store = prepare_vectors(args.paths, device=args.device, verbose=args.verbose)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(vector_store, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(vector_store)} encoded documents to {args.output}")
