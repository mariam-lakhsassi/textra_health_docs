.. role:: red
   :class: red

.. role:: green
   :class: green

.. role:: blue
   :class: blue

.. role:: orange
   :class: orange

.. raw:: html

   <style>
   .red {color: red;}
   .green {color: green;}
   .blue {color: blue;}
   .orange {color: orange;}
   </style>

EAST: Efficient and Accurate Scene Text Detector
===============================================

Introduction
------------

EAST (Efficient and Accurate Scene Text Detector) is a deep learning-based algorithm for detecting text in images. It is designed to localize text regions by directly predicting bounding boxes (either rotated rectangles or quadrilaterals) without the need for intermediate region proposals. EAST is particularly effective for detecting text in natural scenes and scanned documents.

Key Features:
- Detects text of arbitrary orientation.
- Produces rotated or rectangular bounding boxes.
- Fast and accurate, suitable for real-time applications.

Model Architecture
------------------

The EAST model is based on a Fully Convolutional Network (FCN) architecture. It uses a feature extractor (such as VGG16 or ResNet) followed by several convolutional layers to predict:
- **Score Maps**: Indicate the presence of text in different regions.
- **Geometry Maps**: Provide information about the bounding boxes (coordinates, rotation, etc.).

.. image:: image/EAST.png
   :alt: EAST model architecture
   :align: center
...existing code...

Workflow Overview
-----------------

The typical workflow for using EAST in the provided notebook is as follows:

1. **Import Required Libraries**

   .. code-block:: python

      from east import EAST_OUTPUT_LAYERS, decode_predictions
      from imutils import paths
      import numpy as np
      import cv2 as cv
      import matplotlib.pyplot as plt

2. **Set Dataset and Model Paths**

   .. code-block:: python

      dataset = 'dataset'  # Folder path containing images
      image_paths = list(paths.list_images(dataset))
      model_path = 'model/frozen_east_text_detection.pb'

3. **Read and Prepare the Image**

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

7. **Post-process Predictions**

   Use the ``decode_predictions`` function to extract bounding boxes and confidence scores:

   .. code-block:: python

      (rects, confidence) = decode_predictions(scores, geometry)

8. **Apply Non-Maxima Suppression**

   To filter overlapping boxes and keep only the best ones:

   .. code-block:: python

      idxs = cv.dnn.NMSBoxesRotated(rects, confidence, 0.5, 0.4)
      if len(idxs) > 0:
          for i in idxs.flatten():
              box = cv.boxPoints(rects[i])
              box[:, 0] *= rW
              box[:, 1] *= rH
              box = np.int0(box)
              cv.polylines(image, [box], True, (0, 255, 0), 2)

9. **Visualize the Results**

   .. code-block:: python

      plt.figure(figsize=(10,10))
      plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
      plt.show()

Batch Processing
----------------

You can process multiple images in a loop:

.. code-block:: python

   for img_path in image_paths:
       img = cv.imread(img_path)
       # ... (repeat preprocessing, prediction, and visualization steps)

How to Use
----------

1. Place your images in the `dataset` folder.
2. Ensure the pre-trained model file `frozen_east_text_detection.pb` is in the `model` directory.
3. Run the notebook cells in order, or adapt the code into your own Python script.
4. The detected text regions will be highlighted with green bounding boxes on the images.

Tips & Limitations
------------------

- The model works best on reasonably sized and clear text.
- Small or vertical text may not be detected reliably.
- Some false positives or missed detections can occur, especially in challenging conditions.

Conclusion
----------

The EAST model provides an efficient and accurate method for scene text detection. With minimal preprocessing and post-processing, it can be integrated into document analysis or OCR pipelines.

For more details, see the full code and explanations in [`notebook/east/East.ipynb`](notebook/east/East.ipynb).