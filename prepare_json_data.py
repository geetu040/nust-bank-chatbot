import argparse
import json
from config import Q_PROMPT
from utils import clean_text
from tqdm import tqdm

def extract_qa_from_json(path, verbose=False):
    with open(path, 'rb') as f:
        data = json.load(f)

    pairs = {}
    data = data['categories']
    for sub_data in data:
        category = sub_data['category']
        for sub_sub_data in tqdm(sub_data['questions']):
            q = Q_PROMPT.format(heading=category, question=sub_sub_data['question'])
            a = sub_sub_data['answer']
            q = clean_text(q)
            a = clean_text(a)
            pairs[q] = a

    return pairs

"""
python prepare_json_data.py --path "data/funds_transfer_app_features_faq (1).json" --output "processed_data/json_output.json" --verbose
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to JSON file")
    parser.add_argument('--output', type=str, required=True, help="Path to output JSON file")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")
    args = parser.parse_args()

    qa_pairs = extract_qa_from_json(args.path, args.verbose)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"Extracted QA pairs saved to {args.output}")
