#!/usr/bin/env python
# coding: utf-8

# # Text detection with OpenCV (EAST)
# 
# In computer vision, many tasks are devoted to object detection. We might interpret a text as an object in an image. The technique of detecting text in images is a paradigm in the computer vision community due to the enormous efforts in text detection. In real-world problems, images are not perfects and, most of the text detector fails in their tasks.
# 
# As highlighted by [PyImageSearch](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/), when summarizing the paper [Natural Scene Text Understanding](https://www.researchgate.net/publication/221786127_Natural_Scene_Text_Understanding), the common issues are
# 
# - **Image/sensor noise**: Sensor noise from a handheld camera is typically higher than that of a traditional scanner. Additionally, low-priced cameras will typically interpolate the pixels of raw sensors to produce real colors.
# 
# - **Viewing angles**: Natural scene text can naturally have viewing angles that are not parallel to the text, making the text harder to recognize.
# 
# - **Blurring**: Uncontrolled environments tend to have blurred, especially if the end-user is utilizing a smartphone that does not have some form of stabilization.
# 
# - **Lighting conditions**: We cannot make any assumptions regarding our lighting conditions in natural scene images. It may be near dark, the flash on the camera may be on, or the sun may be shining brightly, saturating the entire image.
# 
# - **Resolution**: Not all cameras are created equal — we may be dealing with cameras with sub-par resolution.
# 
# - **Non-paper objects**: Most, but not all, paper is not reflective (at least in the context of paper you are trying to scan). Text in natural scenes may be reflective, including logos, signs, etc.
# 
# - **Non-planar objects**: Consider what happens when you wrap text around a bottle — the text on the surface becomes distorted and deformed. While humans may still be able to easily “detect” and read the text, our algorithms will struggle. We need to be able to handle such use cases.
# 
# - **Unknown layout**: We cannot use any a priori information to give our algorithms “clues” as to where the text resides.
# 
# In the work "[EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)", the authors propose a deep learning-based model for text detection. The idea behind the model is to directly predict words or text lines of arbitrary orientations and quadrilateral shapes in full images, eliminating unnecessary intermediate steps, with a single neural network.
# 
# The model architecture is represented in the following image.
# 
# ![EAST model architecture!](image/EAST.png "EAST model")
# 
# The model consists of two stages: a Fully Convolutional Network (FCN) and an NMS (Nom Maximum Suppression). The FCN can be decomposed into two parts: feature extractor stem and feature-merging. The output layer can offer two different geometries, Rotated Box or Quadrangle. To obtain the final results, the geometries that survived after thresholding will be merged by NMS.
# 
# **Project Structure**
# 
# The project contains three folders. The first of them is the EAST module. This module determines which layers of the network we want to consider, in addition, it offers a pre-process function to convert the predictions into bounding box coordinates. The second folder contains the pre-trained model. The last folder is the dataset used in this example. The images are different scenarios and all of them contain text information.

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


from google.colab import files
uploaded = files.upload()


# ## Importing Libraries
# 

# In[ ]:


# East module
from east import EAST_OUTPUT_LAYERS
from east import decode_predictions
# Standart modules
from imutils import paths
import imutils
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# In[ ]:


get_ipython().system('pip install opencv-python')


# In[ ]:


get_ipython().system('pip install imutils')


# ## Setting the dataset and the pre-trained model path

# In[ ]:


dataset = '/content/drive/MyDrive/S8/textra_health/bounding_box/datset' #folder path
paths = list(paths.list_images(dataset)) # list of image paths


# In[ ]:


model = '/content/drive/MyDrive/S8/textra_health/bounding_box/frozen_east_text_detection.pb'


# ## Reading the test image

# In[ ]:


image = cv.imread(paths[2])


# **Grabbing the original height and width**

# In[ ]:


(origH, origW) = image.shape[:2] #grabbing the image dimensions


# **Defining the new image dimension and aspect ratio**

# In[ ]:


#defining the new Width and height, calculating the aspect ratio
(newW, newH) = (320, 320)
rW = origW / float(newW)
rH = origH / float(newH)


# ## Defining the model

# In[ ]:


net = cv.dnn.readNet(model)


# **Preparing the image for predictions**

# In[ ]:


blob = cv.dnn.blobFromImage(image, 1.0, (newW, newH),
                            (123.68, 116.78, 103.94),
                            swapRB=True, crop=False) #preparing the image for predictions


# In[ ]:


net.setInput(blob)# passing the transformed image


# **making predictions**
# 
# Using the pre-trained model, we want to obtain the confidence score and the geometry informations to derive the text bounding boxes in the image.

# In[ ]:


(scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)


# **Post processing**
# 
# The model gives us the confidences and informations about the possible bounding boxes, but to obtain the exact bounding boxes, we must post-process these informations. To perform it, we call the function `decode_prediction` that returns the confidences and the bounding boxes coordinates. You find the details of the function in the file `east.py`.

# In[ ]:


(rects, confidence) = decode_predictions(scores, geometry)


# **Selecting the bounding boxes**

# In[ ]:


#Non-maxima suppression to obtain on single box
idxs = cv.dnn.NMSBoxesRotated(rects, confidence, 0.5, 0.4)


# In[ ]:


if len(idxs) > 0:

    for i in idxs.flatten():

        box = cv.boxPoints(rects[i])
        box[:, 0] *= rW # rescaling the width
        box[:, 1] *= rH # rescaling the height
        #box = np.int0(box) # putting all bounding boxes coordinates together
        #The correct function to use is either np.int64, np.int32 or np.int_
        box = np.int_(box) #Use np.int_ or another suitable integer type for conversion.
        #This line was already present in your code and seems to be the intended solution.
        cv.polylines(image, [box], True, (0, 255, 0), 2) # drawing the bounding box on the original image


# **Visualizing the prediction**

# In[ ]:


plt.figure(figsize=(10,10))
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.show()


# ## Detecting in multiples images

# In[ ]:


count = 1
plt.figure(figsize=(20,20))
for i in paths:
    img = cv.imread(i)
    # Check if the image was loaded successfully
    if img is not None:
        (origH, origW) = img.shape[:2] #grabbing the image dimensions
        #defining the new Width and height, calculating the aspect ratio
        (newW, newH) = (320, 320)
        rW = origW / float(newW)
        rH = origH / float(newH)

        net = cv.dnn.readNet(model)

        blob = cv.dnn.blobFromImage(img, 1.0, (newW, newH),
                                (123.68, 116.78, 103.94),
                                swapRB=True, crop=False) #preparing the image for predictions

        net.setInput(blob)# passing the transformed image

        (scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)

        (rects, confidence) = decode_predictions(scores, geometry)

        #Non-maxima suppression to obtain on single box
        idxs = cv.dnn.NMSBoxesRotated(rects, confidence, 0.5, 0.4)

        if len(idxs) > 0:

            for i in idxs.flatten():

                box = cv.boxPoints(rects[i])
                box[:, 0] *= rW
                box[:, 1] *= rH
                # box = np.int0(box) # This line caused the error.
                box = np.int_(box) # Use np.int_ or another suitable integer type for conversion.

                cv.polylines(img, [box], True, (0, 255, 0), 2)
        ax = plt.subplot(3,2,count)
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.axis('off')
        count += 1
    else:
        print(f"Failed to load image: {i}")  # Print a message if image loading fails
plt.show()


# ## Conclusion
# 
# The EAST model is an efficient method to detect text in images. The results are near to be perfect, but as with all models, there are some limitations. For example, in the passport, the text with a small size was not detected, beyond it, there's a false positive case and some numbers without detection. Another remark is the number 23 on the t-shirt of the basketball player, the bounding box was shrunken, this is a characteristic of the model. In the image of the French Health Card, the number of series is positioned in the vertical and not detected.

# In[ ]:


def extract_text_regions(image_path, text_detector_model):
    """
    Extract individual word regions from an image using the text detector

    Args:
        image_path: Path to the image
        text_detector_model: Path to the EAST text detection model

    Returns:
        original_image: The original image
        word_regions: List of cropped word images
        boxes: List of bounding boxes coordinates
    """
    # Read the image
    original_image = cv.imread(image_path)
    image = original_image.copy()

    # Get dimensions
    (origH, origW) = image.shape[:2]
    (newW, newH) = (320, 320)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # Load the model
    net = cv.dnn.readNet(text_detector_model)

    # Prepare the image for prediction
    blob = cv.dnn.blobFromImage(image, 1.0, (newW, newH),
                               (123.68, 116.78, 103.94),
                               swapRB=True, crop=False)

    # Set the input and get predictions
    net.setInput(blob)
    (scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)

    # Decode predictions
    (rects, confidence) = decode_predictions(scores, geometry)

    # Apply non-maxima suppression
    idxs = cv.dnn.NMSBoxesRotated(rects, confidence, 0.5, 0.4)

    word_regions = []
    boxes = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get bounding box
            box = cv.boxPoints(rects[i])
            box[:, 0] *= rW
            box[:, 1] *= rH
            box = np.int_(box)

            # Draw bounding box on the image
            cv.polylines(image, [box], True, (0, 255, 0), 2)

            # Extract the word region
            rect = cv.boundingRect(box)
            x, y, w, h = rect

            # Add some padding around the word
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(origW - x, w + 2*padding)
            h = min(origH - y, h + 2*padding)

            # Crop the word region
            word_region = original_image[y:y+h, x:x+w]

            word_regions.append(word_region)
            boxes.append((x, y, w, h))

    return original_image, image, word_regions, boxes

# Now let's create a function that will process a prescription image and predict text
def process_prescription(image_path, text_detector_model):
    """
    Process a prescription image: detect words, predict text for each word,
    and reconstruct the full text
    """
    original_image, annotated_image, word_regions, boxes = extract_text_regions(image_path, text_detector_model)

    # Sort boxes from left to right, top to bottom (reading order)
    # This is a simple sorting method - you might need to refine it based on your layout
    boxes_with_regions = list(zip(boxes, word_regions))

    # Sort by y-coordinate first (row), then by x-coordinate (column)
    # You may need to adjust the threshold for determining if words are on the same line
    same_line_threshold = 20  # pixels

    # Group boxes by line based on y-coordinate
    lines = {}
    for i, (box, _) in enumerate(boxes_with_regions):
        x, y, w, h = box
        line_found = False
        for line_y in lines.keys():
            if abs(y - line_y) < same_line_threshold:
                lines[line_y].append((i, box, word_regions[i]))
                line_found = True
                break
        if not line_found:
            lines[y] = [(i, box, word_regions[i])]

    # Sort each line by x-coordinate
    sorted_lines = []
    for line_y in sorted(lines.keys()):
        sorted_lines.append(sorted(lines[line_y], key=lambda item: item[1][0]))

    # Flatten the list of sorted words
    sorted_words = []
    for line in sorted_lines:
        sorted_words.extend(line)

    # Now sorted_words contains the word regions in reading order

    # Display the original image with bounding boxes
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Detected Words")
    plt.imshow(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))
    plt.show()

    # Display each extracted word
    n_words = len(sorted_words)
    cols = 5
    rows = (n_words // cols) + (1 if n_words % cols else 0)

    plt.figure(figsize=(15, rows * 3))
    for i, (idx, box, word_region) in enumerate(sorted_words):
        plt.subplot(rows, cols, i + 1)
        plt.title(f"Word {i+1}")
        plt.imshow(cv.cvtColor(word_region, cv.COLOR_BGR2RGB))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    return sorted_words


# In[ ]:


filename='/content/drive/MyDrive/S8/textra_health/bounding_box/datset/9.jpg'

import os
print(os.path.exists(filename))  # doit renvoyer True


# In[ ]:


process_prescription(filename, model)


# In[ ]:


# @title Download checkpoint, tokenizer and dataset to local filesystem.
#
import os
import kagglehub

# Use these for PaliGemma-2 3B 224px²
LLM_VARIANT = "gemma2_2b"
MODEL_PATH = "./paligemma2-3b-pt-224.b16.npz"
KAGGLE_HANDLE = "google/paligemma-2/jax/paligemma2-3b-pt-224"  # Path to fetch from Kaggle.

# Use these for PaliGemma 1:
# LLM_VARIANT = "gemma_2b"
# MODEL_PATH = "./paligemma-3b-pt-224.f16.npz"
# KAGGLE_HANDLE = "google/paligemma/jax/paligemma-3b-pt-224"

if not os.path.exists(MODEL_PATH):
  print("Downloading the checkpoint from Kaggle, this could take a few minutes....")
  MODEL_PATH = kagglehub.model_download(KAGGLE_HANDLE, MODEL_PATH)
  print(f"Model path: {MODEL_PATH}")

TOKENIZER_PATH = "./paligemma_tokenizer.model"
if not os.path.exists(TOKENIZER_PATH):
  print("Downloading the model tokenizer...")
  get_ipython().system('gsutil cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}')
  print(f"Tokenizer path: {TOKENIZER_PATH}")

#DATA_DIR="./longcap100"
#if not os.path.exists(DATA_DIR):
 # print("Downloading the dataset...")
  #!gsutil -m -q cp -n -r gs://longcap100/ .
  #print(f"Data path: {DATA_DIR}")


# In[ ]:


import os
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import io


# Add big_vision to path if not already there
if not os.path.exists("big_vision_repo"):
    print("Cloning big_vision repository...")
    os.system("git clone --quiet --branch=main --depth=1 https://github.com/google-research/big_vision big_vision_repo")

if "big_vision_repo" not in sys.path:
    sys.path.append("big_vision_repo")

# Import PaLI-Gemma modules
import jax
import jax.numpy as jnp
import ml_collections
import sentencepiece
import functools
import tensorflow as tf
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
import big_vision.utils
import big_vision.sharding

# Configuration
CHECKPOINT_PATH = '/content/drive/MyDrive/S8/textra_health/my-custom-paligemma-ckpt.npz'
TOKENIZER_PATH = "/content/paligemma_tokenizer.model"
LLM_VARIANT = "gemma2_2b"
SEQLEN = 128
IMAGE_SIZE = 224
EAST_MODEL_PATH = "/content/drive/MyDrive/S8/textra_health/bounding_box/frozen_east_text_detection.pb"  # Path to EAST text detection model

# Set up model configuration
model_config = ml_collections.FrozenConfigDict({
    "llm": {"vocab_size": 257_152, "variant": LLM_VARIANT, "final_logits_softcap": 0.0},
    "img": {"variant": "So400m/14", "pool_type": "none", "scan": True, "dtype_mm": "float16"}
})

# Initialize model and tokenizer
print("Initializing PaLI-Gemma model and tokenizer...")
model = paligemma.Model(**model_config)
tokenizer = sentencepiece.SentencePieceProcessor(TOKENIZER_PATH)

# Define decode function
decode_fn = predict_fns.get_all(model)['decode']
decode = functools.partial(decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id())

def load_params(checkpoint_path):
    """Load model parameters from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Use positional arguments as in the training notebook
    params = paligemma.load(None, checkpoint_path, model_config)

    # Create a pytree mask of the trainable params (for consistency with training)
    def is_trainable_param(name, param):
        if name.startswith("llm/layers/attn/"):  return True
        if name.startswith("llm/"):              return False
        if name.startswith("img/"):              return False
        raise ValueError(f"Unexpected param name {name}")

    trainable_mask = big_vision.utils.tree_map_with_names(is_trainable_param, params)

    # Set up for device distribution
    mesh = jax.sharding.Mesh(jax.devices(), ("data"))
    params_sharding = big_vision.sharding.infer_sharding(
        params, strategy=[('.*', 'fsdp(axis="data")')], mesh=mesh)

    # Handle casting and moving to GPU (same as in training)
    @functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
    def maybe_cast_to_f32(params, trainable):
        return jax.tree.map(lambda p, m: p.astype(jnp.float32)
                          if m else p.astype(jnp.float16),
                          params, trainable)

    # Process params param by param to avoid RAM issues
    params, treedef = jax.tree.flatten(params)
    sharding_leaves = jax.tree.leaves(params_sharding)
    trainable_leaves = jax.tree.leaves(trainable_mask)

    for idx, (sharding, trainable) in enumerate(zip(sharding_leaves, trainable_leaves)):
        params[idx] = big_vision.utils.reshard(params[idx], sharding)
        params[idx] = maybe_cast_to_f32(params[idx], trainable)
        params[idx].block_until_ready()

    params = jax.tree.unflatten(treedef, params)
    print("Checkpoint loaded successfully!")
    return params

def preprocess_image_for_paligemma(cv_image, size=IMAGE_SIZE):
    """Process an OpenCV image for PaLI-Gemma model input."""
    # Convert OpenCV BGR to RGB
    image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)

    # Convert to PIL Image for easier processing
    pil_image = Image.fromarray(image)

    # Convert to numpy array
    image = np.asarray(pil_image)

    # Handle grayscale images
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)

    # Handle RGBA images
    image = image[..., :3]  # Remove alpha channel if present

    # Resize and normalize to [-1, 1]
    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method='bilinear', antialias=True)
    return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]

def preprocess_tokens(prefix, suffix=None, seqlen=None):
    """Tokenize text input for the model."""
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)
    mask_loss = [0] * len(tokens)

    if suffix:
        suffix = tokenizer.encode(suffix, add_eos=True)
        tokens += suffix
        mask_ar += [1] * len(suffix)
        mask_loss += [1] * len(suffix)

    mask_input = [1] * len(tokens)
    if seqlen:
        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding

    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))

def postprocess_tokens(tokens):
    """Convert model output tokens back to text."""
    tokens = tokens.tolist()
    try:
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)

def recognize_word_with_paligemma(cv_image, params, prefix="caption en"):
    """Recognize a single word image with PaLI-Gemma."""
    # Preprocess the image
    image = preprocess_image_for_paligemma(cv_image)

    # Prepare tokens
    tokens, mask_ar, _, mask_input = preprocess_tokens(prefix, seqlen=SEQLEN)

    # Create batch with multiple copies of the same example to match device count
    device_count = jax.device_count()

    # Create a batch by repeating the example to match device count
    batch = {
        "image": np.tile(np.expand_dims(image, axis=0), (device_count, 1, 1, 1)),
        "text": np.tile(np.expand_dims(tokens, axis=0), (device_count, 1)),
        "mask_ar": np.tile(np.expand_dims(mask_ar, axis=0), (device_count, 1)),
        "mask_input": np.tile(np.expand_dims(mask_input, axis=0), (device_count, 1)),
        "_mask": np.ones((device_count,), dtype=bool)  # All examples are real (not padding)
    }

    # Configure sharding for evaluation
    mesh = jax.sharding.Mesh(jax.devices(), ("data"))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    batch = big_vision.utils.reshard(batch, data_sharding)

    # Generate prediction
    tokens = decode({"params": params}, batch=batch, max_decode_len=SEQLEN, sampler="greedy")

    # Process result - just take the first prediction since all are the same
    tokens, mask = jax.device_get((tokens, batch["_mask"]))

    # Get raw prediction from tokens (first example)
    raw_prediction = postprocess_tokens(tokens[0])

    # Remove prefix from prediction
    prefix_with_separator = prefix + "\n"
    if raw_prediction.startswith(prefix_with_separator):
        prediction = raw_prediction[len(prefix_with_separator):]
    else:
        prediction = raw_prediction

    return prediction
def process_prescription_with_paligemma(image_path, east_model_path, paligemma_params):
    """
    Process a prescription image: detect words, recognize each word with PaLI-Gemma,
    and reconstruct the full text
    """
    # Extract word regions using your existing function
    original_image, annotated_image, word_regions, boxes = extract_text_regions(image_path, east_model_path)

    # Sort boxes from left to right, top to bottom (reading order)
    boxes_with_regions = list(zip(boxes, word_regions))

    # Group boxes by line based on y-coordinate
    same_line_threshold = 20  # pixels
    lines = {}
    for i, (box, _) in enumerate(boxes_with_regions):
        x, y, w, h = box
        line_found = False
        for line_y in lines.keys():
            if abs(y - line_y) < same_line_threshold:
                lines[line_y].append((i, box, word_regions[i]))
                line_found = True
                break
        if not line_found:
            lines[y] = [(i, box, word_regions[i])]

    # Sort each line by x-coordinate
    sorted_lines = []
    for line_y in sorted(lines.keys()):
        sorted_lines.append(sorted(lines[line_y], key=lambda item: item[1][0]))

    # Flatten the list of sorted words
    sorted_words = []
    for line in sorted_lines:
        sorted_words.extend(line)

    # Display the original image with bounding boxes
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Detected Words")
    plt.imshow(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))
    plt.show()

    # Process each word with PaLI-Gemma and display results
    recognition_results = []

    print("Recognizing words with PaLI-Gemma...")
    n_words = len(sorted_words)
    cols = 5
    rows = (n_words // cols) + (1 if n_words % cols else 0)

    plt.figure(figsize=(15, rows * 3))

    for i, (idx, box, word_region) in enumerate(sorted_words):
        # Skip empty regions
        if word_region.size == 0:
            recognized_text = ""
        else:
            # Recognize word with PaLI-Gemma
            try:
                recognized_text = recognize_word_with_paligemma(word_region, paligemma_params)
            except Exception as e:
                print(f"Error recognizing word {i+1}: {str(e)}")
                recognized_text = "[ERROR]"

        # Display the word and recognition result
        plt.subplot(rows, cols, i + 1)
        plt.title(f"Word {i+1}: '{recognized_text}'")
        plt.imshow(cv.cvtColor(word_region, cv.COLOR_BGR2RGB))
        plt.axis('off')

        # Add to results
        recognition_results.append({
            "word_index": i + 1,
            "box": box,
            "recognized_text": recognized_text
        })

    plt.tight_layout()
    plt.show()

    # Reconstruct full text by joining recognized words by line
    full_text = []
    current_line = []
    current_line_y = None

    for line in sorted_lines:
        line_text = []
        for i, (idx, box, word_region) in enumerate(line):
            recognized_index = [r["word_index"] for r in recognition_results].index(idx + 1)
            recognized_text = recognition_results[recognized_index]["recognized_text"]
            if recognized_text:  # Skip empty recognitions
                line_text.append(recognized_text)

        full_text.append(" ".join(line_text))

    reconstructed_text = "\n".join(full_text)
    print("\nReconstructed Prescription Text:")
    print(reconstructed_text)

    return {
        "original_image": original_image,
        "annotated_image": annotated_image,
        "word_regions": word_regions,
        "recognition_results": recognition_results,
        "reconstructed_text": reconstructed_text
    }

def main():
    # Load PaLI-Gemma model parameters
    params = load_params(CHECKPOINT_PATH)

    # Path to the prescription image
    prescription_image_path = "/content/drive/MyDrive/S8/textra_health/bounding_box/datset/9.jpg"

    # Process the prescription
    results = process_prescription_with_paligemma(prescription_image_path, EAST_MODEL_PATH, params)

    # Save results if needed
    output_file = "prescription_ocr_results.json"
    with open(output_file, "w") as f:
        # Convert numpy arrays and PIL images to serializable format
        serializable_results = {
            "recognition_results": results["recognition_results"],
            "reconstructed_text": results["reconstructed_text"]
        }
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    print("Starting prescription OCR with PaLI-Gemma...")
    main()


# In[ ]:


get_ipython().system('pip install ml_collections')


# In[ ]:


get_ipython().system('pip install tensorflow')


# In[ ]:




