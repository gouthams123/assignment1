import os
from flask import Flask, request, jsonify
from transformers import pipeline
from data_loader import parse_document

app = Flask(__name__)

print("Loading AI Model for API...")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

QUESTIONS = {
    "Agreement Value": "What is the agreement value or rent amount?",
    "Agreement Start Date": "What is the start date of the agreement?",
    "Agreement End Date": "What is the end date of the agreement?",
    "Renewal Notice (Days)": "What is the renewal notice period in days?",
    "Party One": "Who is the first party, landlord, or owner?",
    "Party Two": "Who is the second party, tenant, or renter?"
}

# Ensure a temporary directory exists for uploads
os.makedirs("uploads", exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # 2. Save the file temporarily
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    try:
        # 3. Extract text using your Day 1 script
        document_text = parse_document(file_path)
        
        # 4. Predict fields using your Day 2 model
        extracted_data = {}
        for field, question in QUESTIONS.items():
            try:
                prediction = qa_pipeline(question=question, context=document_text)
                extracted_data[field] = prediction['answer']
            except Exception:
                extracted_data[field] = "Not Found"
                
        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify({
            "filename": file.filename,
            "metadata": extracted_data
        })
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting API Server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)