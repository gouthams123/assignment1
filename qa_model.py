import json
import pandas as pd
from transformers import pipeline

# Initialize the Hugging Face QA pipeline
# We use a robust model fine-tuned on the SQuAD2 dataset for reading comprehension
print("Loading AI Model (this may take a minute on the first run)...")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define the questions mapped to the exact fields we need to extract
QUESTIONS = {
    "Agreement Value": "What is the agreement value or rent amount?",
    "Agreement Start Date": "What is the start date of the agreement?",
    "Agreement End Date": "What is the end date of the agreement?",
    "Renewal Notice (Days)": "What is the renewal notice period in days?",
    "Party One": "Who is the first party, landlord, or owner?",
    "Party Two": "Who is the second party, tenant, or renter?"
}

def evaluate_predictions(df):
    """Run the QA model and calculate the Per-Field Recall."""
    true_matches = {field: 0 for field in QUESTIONS.keys()}
    false_matches = {field: 0 for field in QUESTIONS.keys()}
    
    print("\nStarting extraction predictions...\n" + "-"*40)
    
    for index, row in df.iterrows():
        context = row.get("document_text", "")
        if not context or pd.isna(context):
            continue
            
        print(f"Processing Document: {row.iloc[0]}")
        
        for field, question in QUESTIONS.items():
            # Get the true ground truth value from the dataframe
            # Adjust 'field' if your CSV column names differ slightly
            ground_truth = str(row.get(field, "")).strip().lower()
            
            # Predict the answer using the AI model
            try:
                prediction = qa_pipeline(question=question, context=context)
                predicted_answer = str(prediction['answer']).strip().lower()
            except Exception:
                predicted_answer = ""
                
            # Exact Match Evaluation
            if ground_truth and ground_truth in predicted_answer or predicted_answer in ground_truth:
                true_matches[field] += 1
            else:
                false_matches[field] += 1
                
    # Calculate and display Per-Field Recall
    print("\n" + "="*40 + "\nEvaluation Results (Per-Field Recall)\n" + "="*40)
    for field in QUESTIONS.keys():
        t = true_matches[field]
        f = false_matches[field]
        recall = t / (t + f) if (t + f) > 0 else 0.0
        print(f"{field}: True={t}, False={f} --> Recall = {recall:.2f}")

if __name__ == "__main__":
    # Load the unified data we created yesterday
    try:
        df_train = pd.read_json("processed_train.json")
        evaluate_predictions(df_train)
    except FileNotFoundError:
        print("Error: processed_train.json not found. Please run data_loader.py first.")