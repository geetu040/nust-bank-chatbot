from flask import Flask, request, jsonify, render_template
from chatbot import Chatbot, ChatbotConfig
import PyPDF2
from io import BytesIO

app = Flask(__name__)

# Initialize your existing chatbot
config = ChatbotConfig(
    chatbot_model_name="Qwen/Qwen3-0.6B",
    use_cache=True
)
chatbot = Chatbot(config)

@app.route('/')
def home():
    return render_template('home.html')  

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    answer, meta = chatbot.query(question)
    return jsonify({
        'answer': answer,
        'sources': meta['relevant_docs']
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    print("hees")
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    for file in files:
        if file.filename.endswith('.pdf'):
            # Extract text from PDF
            pdf_file = BytesIO(file.read())
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = "\n\n".join([page.extract_text() for page in pdf_reader.pages])
            
            # Add to chatbot's knowledge
            print(text)
            chatbot.add_document(text)
    
    return jsonify({'status': 'success', 'message': f'Processed {len(files)} files'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)