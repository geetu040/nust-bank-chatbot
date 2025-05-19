
# Setting Up

```cmd
pip install -r requirements.txt
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
