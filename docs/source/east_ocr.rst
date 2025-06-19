OCR Pipeline: Text Detection with EAST and Recognition with Tesseract
====================================================================

This documentation describes a complete OCR (Optical Character Recognition) pipeline using the EAST text detector and Tesseract OCR engine. The pipeline first detects text regions in images using the pre-trained EAST model, then extracts and recognizes the text using Tesseract.

Introduction
------------

Text detection and recognition in natural images is a challenging task due to noise, varying lighting, perspective distortions, and complex backgrounds. The EAST (Efficient and Accurate Scene Text Detector) model provides a robust solution for detecting text regions of arbitrary orientation. Once detected, these regions can be cropped and passed to Tesseract for text recognition.

Project Structure
-----------------

- ``east/``  
  - ``east.py``: Contains helper functions and constants for the EAST model.
  - ``model/``: Contains the pre-trained ``frozen_east_text_detection.pb`` model.
  - ``image/``: Example images for testing.
- ``Simple OCR using East and pytesseract (1).ipynb``: Notebook demonstrating the full OCR pipeline.
- ``East.ipynb``: Notebook focused on text detection with EAST.

Workflow Overview
-----------------

1. **Import Required Libraries**

   .. code-block:: python

      from east import EAST_OUTPUT_LAYERS, decode_predictions
      from imutils import paths
      import numpy as np
      import cv2 as cv
      import matplotlib.pyplot as plt
      import pytesseract

2. **Set Paths**

   .. code-block:: python

      dataset = 'dataset'  # Folder containing images
      image_paths = list(paths.list_images(dataset))
      model_path = 'model/frozen_east_text_detection.pb'

3. **Load and Prepare Image**

   .. code-block:: python

      image = cv.imread(image_paths[1])
      (origH, origW) = image.shape[:2]
      (newW, newH) = (320, 320)
      rW = origW / float(newW)
      rH = origH / float(newH)

4. **Load the Pre-trained EAST Model**

   .. code-block:: python

      net = cv.dnn.readNet(model_path)

5. **Preprocess the Image**

   .. code-block:: python

      blob = cv.dnn.blobFromImage(image, 1.0, (newW, newH),
                                  (123.68, 116.78, 103.94),
                                  swapRB=True, crop=False)
      net.setInput(blob)

6. **Make Predictions**

   .. code-block:: python

      (scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)

7. **Decode Predictions**

   .. code-block:: python

      (rects, confidence) = decode_predictions(scores, geometry)

8. **Apply Non-Maxima Suppression**

   .. code-block:: python

      idxs = cv.dnn.NMSBoxesRotated(rects, confidence, 0.5, 0.4)
      boxes = []
      if len(idxs) > 0:
          for i in idxs.flatten():
              box = cv.boxPoints(rects[i])
              box[:, 0] *= rW
              box[:, 1] *= rH
              box = np.int0(box)
              boxes.append(box)
              cv.polylines(image, [box], True, (0, 255, 0), 2)

9. **Find the Smallest Bounding Box Covering All Detected Boxes**

   .. code-block:: python

      all_points = np.vstack(boxes)
      x, y, w, h = cv.boundingRect(all_points)
      cropped = image[y:y+h, x:x+w]

10. **Recognize Text with Tesseract**

    .. code-block:: python

       import pytesseract
       text = pytesseract.image_to_string(cropped)
       print("Recognized Text:", text)

Batch Processing
----------------

To process multiple images:

.. code-block:: python

   for img_path in image_paths:
       # Repeat steps 3-10 for each image
       ...

Tips & Limitations
------------------

- The EAST model is robust for horizontal and rotated text, but may miss very small or vertical text.
- Preprocessing (e.g., resizing, denoising) can improve detection and recognition.
- Tesseract works best on clean, high-contrast crops.

Example Output
--------------

- Detected text regions are highlighted with green bounding boxes.
- The cropped region containing all detected text is passed to Tesseract for recognition.
- The recognized text is printed to the console.

References
----------

- `EAST: An Efficient and Accurate Scene Text Detector <https://arxiv.org/abs/1704.03155>`_
- `PyImageSearch EAST Tutorial <https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/>`_
- `Tesseract OCR <https://github.com/tesseract-ocr/tesseract>`_

See Also
--------

- :doc:`East <east>`
- :doc:`notebook/Simple OCR using East and pytesseract (1).ipynb <../../notebook/Simple OCR using East and pytesseract (1)>`