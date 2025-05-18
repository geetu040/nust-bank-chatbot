import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import clean_text


class ChatbotConfig:
	vector_store_path = 'processed_data/vector_store.json'
	embedding_model_name = 'all-MiniLM-L6-v2'
	chatbot_model_name = 'google/flan-t5-base'
	device = 'cpu'


class Chatbot:
	def __init__(self, config: ChatbotConfig):
		self.config = config
		self.init()


	def init(self):
		self.docs, self.index = self.load_vector_store()
		self.embedding_model = self.load_embedding_model()
		self.tokenizer, self.llm = self.load_qa_model()


	def load_embedding_model(self):
		model = SentenceTransformer(
			self.config.embedding_model_name,
			device=self.config.device
		)
		return model


	def load_vector_store(self):
		with open(self.config.vector_store_path, 'rb') as f:
			data = json.load(f)

		docs = data['docs']
		vectors = np.array(data['vectors'])
		index = faiss.IndexFlatIP(vectors.shape[1])
		index.add(vectors)
		return docs, index


	def load_qa_model(self):
		tokenizer = AutoTokenizer.from_pretrained(self.config.chatbot_model_name)
		model = AutoModelForSeq2SeqLM.from_pretrained(self.config.chatbot_model_name)
		return tokenizer, model


	def query(self, question):

		# Step 1: Preprocess the question
		question = clean_text(question)

		# Step 2: Encode the question
		question_vector = self.embedding_model.encode([question])

		# Step 3: Search for relevant documents
		D, I = self.index.search(question_vector, k=5)
		relevant_docs = []
		for i in I[0]:
			if i < len(self.docs):
				relevant_docs.append(self.docs[i])

		# Step 4: Create the prompt
		context = "\n\n".join(relevant_docs)
		prompt = (
			f"You are a helpful and professional bank assistant.\n"
			f"When the question is about branches, complaints, or services, include any available details like contact information or process steps.\n"
			f"Use the following context to answer the question clearly and completely.\n\n"
			f"Context:\n{context}\n\n"
			f"Question: {question}\n"
			f"Answer:"
		)

		# Step 5: Generate answer
		input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
		output_ids = self.llm.generate(
			input_ids,
			# max_new_tokens=200,
		)

		# Step 6: Decode response
		generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
		answer = generated_text
		# answer = generated_text.split("Answer:")[-1].strip() if "Answer:" in generated_text else generated_text.strip()

		# Step 7: Return response
		meta = {
			"relevant_docs": relevant_docs,
		}
		return answer, meta


if __name__ == "__main__":
	config = ChatbotConfig()
	chatbot = Chatbot(config)

	question = "How much loan can i avail?"
	answer, meta = chatbot.query(question)

	print()
	print()
	print()
	print('Question:')
	print(question)
	print()
	print('Meta:')
	print(meta)
	print()
	print('Answer:')
	print(answer)

