OCR with EAST and DocLing
=========================

Overview
--------

This documentation describes how to combine the EAST text detector with the DocLing OCR pipeline for robust text extraction from images and PDFs. The approach leverages deep learning for text region detection and state-of-the-art OCR models for accurate recognition, making it suitable for complex documents such as medical prescriptions.

Pipeline Steps
--------------

1. **Mount Google Drive (Colab only)**

   If running in Google Colab, mount your Google Drive to access files:

   .. code-block:: python

      from google.colab import drive
      drive.mount('/content/drive')

2. **Install Required Packages**

   Install DocLing and RapidOCR:

   .. code-block:: python

      !pip install docling
      !pip install rapidocr_onnxruntime

3. **Import Libraries**

   .. code-block:: python

      import os
      import cv2 as cv
      import matplotlib.pyplot as plt
      import numpy as np
      import copy
      import math
      from huggingface_hub import snapshot_download
      from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
      from docling.document_converter import (
          ConversionResult, DocumentConverter, InputFormat, PdfFormatOption,
      )

4. **EAST Text Detection**

   The EAST model is used to detect text regions in the image. The `decode` and `get_EAST_regions` functions process the model output and extract bounding boxes for text areas.

   .. code-block:: python

      def get_EAST_regions(image):
          # ... (see notebook for full code)
          net = cv.dnn.readNet(PATH_TO_MODEL)
          # Preprocess and run model
          # Decode predictions and apply NMS
          return f_boxes  # List of bounding boxes

5. **Draw and Aggregate Detected Regions**

   Draw detected bounding boxes and compute a single bounding box that contains all detected text regions:

   .. code-block:: python

      def detect_text_region(image):
          east_boxes = get_EAST_regions(image)
          # Draw boxes and compute min/max coordinates
          return x_min, y_min, x_max, y_max

6. **Set Up DocLing OCR**

   Download RapidOCR models and configure DocLing for OCR:

   .. code-block:: python

      def setup_docling_ocr():
          download_path = snapshot_download(repo_id="SWHL/RapidOCR")
          # Set up model paths and options
          converter = DocumentConverter(
              format_options={
                  InputFormat.PDF: PdfFormatOption(
                      pipeline_options=pipeline_options,
                  ),
              },
          )
          return converter

7. **Recognize Text with DocLing**

   Save the detected text region as a temporary PDF and use DocLing to extract text:

   .. code-block:: python

      def recognize_image_with_docling(image, converter):
          # Save image as PDF
          # Use converter.convert() to process PDF
          # Extract and return recognized text lines

8. **Full Image Processing Pipeline**

   The main function combines all steps: detection, cropping, OCR, and output.

   .. code-block:: python

      def process_image_with_east_docling(image_path):
          image = cv.imread(image_path)
          image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
          converter = setup_docling_ocr()
          x_min, y_min, x_max, y_max = detect_text_region(image)
          text_region = image[y_min:y_max, x_min:x_max]
          lines = recognize_image_with_docling(text_region, converter)
          for line in lines:
              print(line)

9. **PDF Processing (Optional)**

   DocLing can also process PDFs directly without EAST:

   .. code-block:: python

      def process_pdf_with_docling(pdf_path):
          # Setup DocLing and process PDF
          # Save recognized text as Markdown

Usage Example
-------------

To process an image:

.. code-block:: python

   image_file = "/path/to/image.jpg"
   process_image_with_east_docling(image_file)

To process a PDF:

.. code-block:: python

   pdf_file = "/path/to/document.pdf"
   process_pdf_with_docling(pdf_file)

Key Points
----------

- **EAST** is used to localize text regions, improving OCR accuracy by focusing on relevant areas.
- **DocLing** (with RapidOCR) provides robust text recognition, supporting both images and PDFs.
- The pipeline is modular: you can use only OCR, or combine detection and OCR for better results on complex layouts.

References
----------

- `DocLing documentation <https://github.com/baidu-research/docling>`_
- `RapidOCR <https://github.com/RapidAI/RapidOCR>`_
- `EAST: An Efficient and Accurate Scene Text Detector <https://arxiv.org/abs/1704.03155>`_
