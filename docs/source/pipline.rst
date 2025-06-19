Textra Health: End-to-End Prescription Analysis Pipeline
========================================================

Overview
--------

This documentation describes the end-to-end pipeline for analyzing medical prescriptions and generating detailed reports on prescribed medications using OCR, deep learning, and large language models (LLMs).

Pipeline Architecture
---------------------

.. image:: ../images/pipeline_diagram.png
   :alt: Pipeline architecture diagram
   :align: center

**Pipeline Steps:**

1. **Input: Image d'une prescription**
   
   The process begins with an input image of a medical prescription, typically captured via a scanner or smartphone.

2. **Prétraitement (Preprocessing)**
   
   The input image undergoes preprocessing to enhance quality and improve text detection. This may include resizing, denoising, contrast adjustment, and binarization.

3. **EAST (Text Detection)**
   
   The preprocessed image is passed through the EAST (Efficient and Accurate Scene Text Detector) model to localize text regions, even if they are rotated or at arbitrary angles.

4. **OCR (Text Recognition with docling)**
   
   Detected text regions are extracted and recognized using an OCR engine (such as Tesseract, wrapped in the docling library). This step converts image regions into machine-readable text.

5. **LLM (Llama 3.2:3b)**
   
   The recognized text is sent to a Large Language Model (LLM), such as Llama 3.2:3b, for further analysis. The LLM is responsible for understanding the prescription content and generating a structured report.

6. **Medical Data Processing**
   
   - **Data médicale:** The system uses a medical knowledge base containing information about diseases, medications, general drug information, and medical terminology.
   - **Chunks:** The medical data is divided into manageable chunks.
   - **Embedding:** Each chunk is converted into vector representations (embeddings).
   - **Vector database:** Embeddings are stored in a vector database for efficient retrieval and context augmentation during LLM inference.

7. **Output: Rapport sur les médicaments prescrits**
   
   The final output is a detailed report on the prescribed medications, including relevant medical information, usage instructions, and potential interactions.

Key Technologies
----------------

- **EAST:** Deep learning-based text detector for robust scene text localization.
- **OCR (docling):** Optical Character Recognition for converting detected text regions into digital text.
- **LLM (Llama 3.2:3b):** Advanced language model for understanding and summarizing prescription content.
- **Vector Database:** Enables semantic search and retrieval of relevant medical knowledge.

Use Cases
---------

- Automated extraction and analysis of handwritten or printed medical prescriptions.
- Generation of patient-friendly medication reports.
- Support for pharmacists and healthcare professionals in verifying prescriptions.

Summary
-------

This pipeline combines state-of-the-art computer vision and natural language processing to automate the extraction, understanding, and reporting of medical prescription data, improving efficiency and reducing errors in healthcare workflows.