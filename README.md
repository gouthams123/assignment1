# Document AI: Metadata Extraction System

This project implements an AI/ML-based system to automatically extract key metadata from legal documents and scanned images. The system supports `.docx` documents and `.png` scanned images.

The solution uses OCR and an Extractive Question Answering (QA) approach to identify the required metadata fields from documents. It avoids rule-based approaches such as RegEx and static conditions, allowing the system to work with different document templates.

## Key Features

- **Multimodal Document Processing:** Supports both `.docx` documents and scanned `.png` images.
- **AI/ML-Based Extraction:** Uses Natural Language Processing and Extractive Question Answering instead of static rules.
- **OCR Support:** Uses Tesseract OCR to extract text from scanned document images.
- **RoBERTa QA Model:** Uses the Hugging Face `roberta-base-squad2` model for extracting answers from document text.
- **Multiple Metadata Fields:** Extracts six important fields from the documents.
- **RESTful API:** The extraction system is exposed through a Flask REST API.
- **Template Independent:** The system is designed to extract information irrespective of the document template format.

## Metadata Fields

The system extracts the following fields:

1. Agreement Value
2. Agreement Start Date
3. Agreement End Date
4. Renewal Notice (Days)
5. Party One
6. Party Two

## Technology Stack

- **Programming Language:** Python 3.10
- **Machine Learning / NLP:** Hugging Face Transformers, PyTorch
- **OCR:** Tesseract OCR, Pytesseract
- **Image Processing:** Pillow
- **Document Processing:** python-docx
- **Data Processing:** Pandas
- **Web Framework:** Flask
- **API Testing:** Requests

## Project Structure

```text
assignment-1/
│
├── data/
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── test.csv
│
├── uploads/
│
├── app.py
├── data_loader.py
├── qa_model.py
├── test_api.py
├── processed_train.json
├── assignment-details.pdf
└── README.md
````

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/gouthams123/assignment1
cd assignment-1
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Project

### Phase 1: Data Ingestion & Alignment

Parses raw documents and images from the `data/train/` directory, extracts the text, and aligns it with the ground-truth metadata in `train.csv`.

```bash
python data_loader.py
```

### Phase 2: Model Evaluation

Evaluates the Hugging Face QA model against the processed dataset, calculating exact-match per-field recall.

```bash
python qa_model.py
```

### Phase 3: Launching the API Server

Wraps the extraction pipeline in a Flask web service.

```bash
python app.py
```

### Phase 4: Testing the API

Open a new, separate terminal window while keeping the API server running. Then run the test script to send a document to the inference endpoint.

```bash
python test_api.py
```

## API Test Output

```json
{
  "filename": "95980236-Rental-Agreement.png",
  "metadata": {
    "Agreement End Date": "1* April 2010",
    "Agreement Start Date": "1‘ May 2005",
    "Agreement Value": "Rs. 9,000/-",
    "Party One": "the Lessor",
    "Party Two": "LESSEE",
    "Renewal Notice (Days)": "one month"
  }
}
```

```
```
