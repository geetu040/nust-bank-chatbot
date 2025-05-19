
# Setting Up

```cmd
pip install -r requirements.txt
```

# Data Preparation

1. create a directory for processed data
```cmd
mkdir processed_data
```

2. prepare excel data
```cmd
python prepare_excel_data.py --path "data/NUST Bank-Product-Knowledge.xlsx" --output "processed_data/excel_output.json" --verbose
```

3. prepare json data
```cmd
python prepare_json_data.py --path "data/funds_transfer_app_features_faq (1).json" --output "processed_data/json_output.json" --verbose
```

4. prepare vector store
```cmd
python prepare_vectors.py --paths processed_data/excel_output.json processed_data/json_output.json --output processed_data/vector_store.json --verbose
```

# Usage

```py
from chatbot import ChatbotConfig, Chatbot

config = ChatbotConfig()
chatbot = Chatbot(config)

question = "How much loan can I avail?"
answer, meta = chatbot.query(question)
print(answer)

new_doc = """What are the free services associated with PakWatan Remittance account?
Free services include:
- First Cheque Book of 25 Leaves*
- NUST Visa Debit Card Issuance* (annual and replacement fee would apply)
- Bankers Cheque Issuance
- Branch Online Cash Withdrawal/ Deposit (Online)
- Fund Transfer within NUST via branch (Online Transfer)
- Internet Banking
- SMS on digital transactions
- E-statements
* For Current Account only
"""
chatbot.add_document(new_doc)

question = "Which accounts are eligible for free services of PWRA?"
answer, meta = chatbot.query(question)
print(answer)
```
