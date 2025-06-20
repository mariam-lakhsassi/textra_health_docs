OCR Pipeline with EAST and PaLI-Gemma VLM
=========================================

This documentation explains step-by-step how to use the EAST text detector in combination with the PaLI-Gemma Vision-Language Model (VLM) for advanced OCR on prescription images. It covers environment setup, model upload, fine-tuning, Hugging Face secrets, and the integration of both models.

Overview
--------

The pipeline detects text regions in a prescription image using the EAST model, then recognizes each word using the PaLI-Gemma VLM. This approach leverages deep learning for both detection and recognition, enabling robust OCR even on complex or handwritten documents.

Environment Setup
-----------------

1. **Install Required Packages**

   .. code-block:: python

      !pip install ml_collections
      !pip install tensorflow

2. **Upload Files in Colab**

   If running in Google Colab, upload your files:

   .. code-block:: python

      from google.colab import files
      uploaded = files.upload()

3. **Mount Google Drive (Colab)**

   .. code-block:: python

      from google.colab import drive
      drive.mount('/content/drive')

Model Preparation
-----------------

### 1. EAST Text Detector

- The EAST model is used to detect text regions in the image.

   .. code-block:: python

      from east import EAST_OUTPUT_LAYERS, decode_predictions

- Example of checking if your image file exists:

   .. code-block:: python

      filename = '/content/drive/MyDrive/S8/textra_health/bounding_box/datset/9.jpg'
      import os
      print(os.path.exists(filename))  # Should return True

### 2. PaLI-Gemma VLM

- Download the PaLI-Gemma model checkpoint and tokenizer. You can use KaggleHub or Hugging Face.

   .. code-block:: python

      import os
      import kagglehub

      LLM_VARIANT = "gemma2_2b"
      MODEL_PATH = "./paligemma2-3b-pt-224.b16.npz"
      KAGGLE_HANDLE = "google/paligemma-2/jax/paligemma2-3b-pt-224"

      if not os.path.exists(MODEL_PATH):
          print("Downloading the checkpoint from Kaggle...")
          MODEL_PATH = kagglehub.model_download(KAGGLE_HANDLE, MODEL_PATH)
          print(f"Model path: {MODEL_PATH}")

      TOKENIZER_PATH = "./paligemma_tokenizer.model"
      if not os.path.exists(TOKENIZER_PATH):
          print("Downloading the model tokenizer...")
          !gsutil cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}
          print(f"Tokenizer path: {TOKENIZER_PATH}")

- **Using Hugging Face secrets**:  
  If you want to download from Hugging Face, you need to set your token as a secret in your environment or notebook settings.  
  Example (Colab):

   .. code-block:: python

      from huggingface_hub import login
      login(token="YOUR_HF_TOKEN")

      # Then use snapshot_download or hf_hub_download as needed

- **Fine-tuning**:  
  To use a fine-tuned model, upload your custom checkpoint (e.g., `my-custom-paligemma-ckpt.npz`) to your drive or workspace and set the path accordingly.

   .. code-block:: python

      CHECKPOINT_PATH = '/content/drive/MyDrive/S8/textra_health/my-custom-paligemma-ckpt.npz'

Pipeline Steps
--------------

### 1. Extract Text Regions with EAST

   .. code-block:: python

      def extract_text_regions(image_path, text_detector_model):
          # Loads image, runs EAST, returns word regions and bounding boxes
          # See notebook for full code

### 2. Process Prescription Image

   .. code-block:: python

      def process_prescription(image_path, text_detector_model):
          # Detects words, sorts them, displays results
          # Returns sorted word regions for recognition

### 3. Load and Initialize PaLI-Gemma

   .. code-block:: python

      import ml_collections
      import sentencepiece
      from big_vision.models.proj.paligemma import paligemma

      model_config = ml_collections.FrozenConfigDict({
          "llm": {"vocab_size": 257_152, "variant": LLM_VARIANT, "final_logits_softcap": 0.0},
          "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
      })

      model = paligemma.Model(**model_config)
      tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

      # Load parameters from checkpoint
      params = load_params(CHECKPOINT_PATH)

### 4. Recognize Words with PaLI-Gemma

   .. code-block:: python

      def recognize_word_with_paligemma(cv_image, params, prefix="caption en"):
          # Preprocess image, tokenize, run model, decode output
          # Returns recognized text

### 5. Full Pipeline: EAST + PaLI-Gemma

   .. code-block:: python

      def process_prescription_with_paligemma(image_path, east_model_path, paligemma_params):
          # Detects words with EAST, recognizes each with PaLI-Gemma, reconstructs full text
          # Displays results and saves output

      # Example usage:
      results = process_prescription_with_paligemma(
          "/content/drive/MyDrive/S8/textra_health/bounding_box/datset/9.jpg",
          EAST_MODEL_PATH,
          params
      )

      print(results["reconstructed_text"])

How to Use
----------

1. **Upload your prescription image and models to your workspace or Google Drive.**
2. **Set the correct paths for the image, EAST model, and PaLI-Gemma checkpoint/tokenizer.**
3. **Run the pipeline as shown above.**
4. **If using Hugging Face or Kaggle, authenticate and download the models as needed.**
5. **For fine-tuned models, simply point to your custom checkpoint.**

Tips
----

- Make sure your environment (Colab, local, etc.) has enough RAM and GPU for PaLI-Gemma.
- Use secrets or environment variables to securely store Hugging Face tokens.
- You can adapt the pipeline to process batches of images or integrate with a web interface.

References
----------

- `EAST: An Efficient and Accurate Scene Text Detector <https://arxiv.org/abs/1704.03155>`_
- `PaLI-Gemma (Google Research) <https://github.com/google-research/big_vision>`_
- `Hugging Face Hub <https://huggingface.co/docs/huggingface_hub/>`_
- `KaggleHub <https://github.com/Kaggle/kagglehub>`_
