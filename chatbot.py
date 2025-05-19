import json
import faiss
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from utils import clean_text
from prepare_excel_data import extract_qa_from_excel
from prepare_json_data import extract_qa_from_json
from utils import prepare_document


class ChatbotConfig:
	def __init__(
		self,
		excel_data_path="data/NUST Bank-Product-Knowledge.xlsx",
		json_data_path="data/funds_transfer_app_features_faq (1).json",
		output_dir='temp/',
		embedding_model_name='all-MiniLM-L6-v2',
		chatbot_model_name='Qwen/Qwen3-0.6B',
		device='cpu',
		use_cache=True,
		enable_thinking=False,
		top_k=3,
	):
		self.excel_data_path = excel_data_path
		self.json_data_path = json_data_path
		self.output_dir = output_dir
		self.embedding_model_name = embedding_model_name
		self.chatbot_model_name = chatbot_model_name
		self.device = device
		self.use_cache = use_cache
		self.enable_thinking = enable_thinking
		self.top_k = top_k

		os.makedirs(self.output_dir, exist_ok=True)


class Chatbot:
	def __init__(self, config: ChatbotConfig):
		self.config = config
		self.init()


	def init(self):
		self.embedding_model = SentenceTransformer(
			self.config.embedding_model_name,
			device=self.config.device
		)
		self.tokenizer = AutoTokenizer.from_pretrained(self.config.chatbot_model_name)
		try:
			self.llm = AutoModelForCausalLM.from_pretrained(self.config.chatbot_model_name)
		except:
			self.llm = AutoModelForSeq2SeqLM.from_pretrained(self.config.chatbot_model_name)

		self.init_vector_store()


	def init_vector_store(self):

		docs_path = os.path.join(self.config.output_dir, "docs.json")
		if os.path.exists(docs_path) and self.config.use_cache:
			with open(docs_path, 'r') as f:
				self.docs = json.load(f)
		else:
			qa_pairs = {} # Dict{question: answer}

			if self.config.excel_data_path is not None:
				excel_data = extract_qa_from_excel(self.config.excel_data_path, verbose=False)
				qa_pairs.update(excel_data)

			if self.config.json_data_path is not None:
				json_data = extract_qa_from_json(self.config.json_data_path, verbose=False)
				qa_pairs.update(json_data)

			self.docs = []
			for q, a in qa_pairs.items():
				doc = prepare_document(q, a)
				self.docs.append(doc)

			with open(docs_path, 'w') as f:
				json.dump(self.docs, f)

		index_path = os.path.join(self.config.output_dir, "index.faiss")
		if os.path.exists(index_path) and self.config.use_cache:
			self.index = faiss.read_index(index_path)
		else:
			vectors = self.embedding_model.encode(self.docs)
			self.index = faiss.IndexFlatIP(vectors.shape[1])
			self.index.add(vectors)
			faiss.write_index(self.index, index_path)


	def query(self, question):

		# Step 1: Preprocess the question
		question = clean_text(question)

		# Step 2: Encode the question
		question_vector = self.embedding_model.encode([question])

		# Step 3: Search for relevant documents
		D, I = self.index.search(question_vector, k=self.config.top_k)
		relevant_docs = []
		for i in I[0]:
			if i < len(self.docs):
				relevant_docs.append(self.docs[i])

		# Step 4: Create the prompt
		context = "\n\n".join(relevant_docs)
		system_prompt = (
			"You are a helpful, caring and knowledgeable assistant for a bank customer service. "
			"Answer questions clearly, concisely, and using the provided context. "
			"Yours answers should be well formated and easy to read. "
			"If the answer is not in the context, say you don't know.\n\n"
			"Context:\n" + context + "\n\n"
		)
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": question}
		]
		try:
			prompt = self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
				enable_thinking=self.config.enable_thinking # Switches between thinking and non-thinking modes. Default is True.
			)
		except:
			prompt = system_prompt + question


		# Step 5: Generate answer
		input_ids = self.tokenizer([prompt], return_tensors="pt", truncation=True).input_ids
		output_ids = self.llm.generate(
			input_ids,
			max_new_tokens=1000,
		)

		# Step 6: Decode response
		answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
		if "think" in answer:
			answer = answer[answer.index("</think>")+8:].strip()

		# Step 7: Return response
		meta = {
			"prompt": prompt,
			"relevant_docs": relevant_docs,
			"relevant_docs_scores": D[0].tolist(),
		}
		return answer, meta


if __name__ == "__main__":
	config = ChatbotConfig(
		chatbot_model_name="Qwen/Qwen3-0.6B",
		# chatbot_model_name="google/flan-t5-base",
		# chatbot_model_name="microsoft/bitnet-b1.58-2B-4T",
		use_cache=False,
	)
	chatbot = Chatbot(config)

	# question = "What are the available Liability Products & Services?"
	# question = "What is NSA?"
	question = "What does PWRA stand for?"
	# question = "What are the posssible account types in NSA?"
	answer, meta = chatbot.query(question)

	print()
	print('------- Question -------')
	print(question)
	print()
	for k, v in meta.items():
		print("-------", k, "-------")
		print(v)
		print()
	print('------- Answer -------')
	print(answer)
