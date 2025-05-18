
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
```
