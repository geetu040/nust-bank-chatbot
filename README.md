
# Setting Up

```cmd
pip install -r requirements.txt
```

# Data Preparation

1. prepare excel data
```py
python prepare_excel_data.py --path "data/NUST Bank-Product-Knowledge.xlsx" --output "processed_data/excel_output.json" --verbose
```

2. prepare json data
```py
python prepare_json_data.py --path "data/funds_transfer_app_features_faq (1).json" --output "processed_data/json_output.json" --verbose
```

3. prepare vector store
```py
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
