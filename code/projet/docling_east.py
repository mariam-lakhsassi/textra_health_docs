import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import sys

from huggingface_hub import snapshot_download
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import (
    ConversionResult, DocumentConverter, InputFormat, PdfFormatOption,
)

# Path to EAST text detection model
PATH_TO_MODEL = "./frozen_east_text_detection.pb"

# decode prediction for EAST model
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):
        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = (
                [offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

# get text bounding boxes using EAST
def get_EAST_regions(image):
    text_region_width = 1000
    confThreshold = 0.1
    nmsThreshold = 0.3
    inpWidth = 1600
    inpHeight = 1280
    model = PATH_TO_MODEL
    net = cv.dnn.readNet(model)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    outNames = []
    outNames.append("feature_fusion/Conv_7/Sigmoid")
    outNames.append("feature_fusion/concat_3")

    image = image.astype(np.uint8)
    height_ = image.shape[0]
    width_ = image.shape[1]
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    # Create a 4D blob from image.
    blob = cv.dnn.blobFromImage(image, 1.0, (inpWidth, inpHeight))
    # Run the model
    net.setInput(blob)
    outs = net.forward(outNames)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

    # Get scores and geometry
    scores = outs[0]
    geometry = outs[1]
    [boxes, confidences] = decode(scores, geometry, confThreshold)
    f_boxes = []

    # Apply NMS
    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

    # Debug info
    print(f"Type of indices: {type(indices)}")
    print(f"Shape/length of indices: {indices.shape if hasattr(indices, 'shape') else len(indices) if hasattr(indices, '__len__') else 'unknown'}")
    if len(indices) > 0:
        print(f"Type of first element: {type(indices[0])}")
        print(f"Value of first element: {indices[0]}")

    # Handle the indices safely regardless of OpenCV version
    f_boxes = []
    try:
        if isinstance(indices, np.ndarray) and indices.size > 0:
            # For OpenCV 4.x
            for idx in indices.flatten():
                # get 4 corners of the rotated rect
                vertices = cv.boxPoints(boxes[idx])
                # scale the bounding box coordinates based on the respective ratios
                for j in range(4):
                    vertices[j][0] *= rW
                    vertices[j][1] *= rH
                f_boxes.append(vertices)
        elif len(indices) > 0:
            # For older OpenCV versions or other return formats
            for idx in indices:
                # Handle both possible formats
                if isinstance(idx, (list, np.ndarray)):
                    idx = idx[0]
                # get 4 corners of the rotated rect
                vertices = cv.boxPoints(boxes[idx])
                # scale the bounding box coordinates based on the respective ratios
                for j in range(4):
                    vertices[j][0] *= rW
                    vertices[j][1] *= rH
                f_boxes.append(vertices)
    except Exception as e:
        print(f"Error processing indices: {e}")
        print(f"Debug info - indices: {indices}")
        if len(indices) > 0:
            print(f"First element: {indices[0]}, Type: {type(indices[0])}")

    return f_boxes

# draw bounding boxes
def draw_detected_bounding_boxes(image, east_boxes):
    for box in east_boxes:
        x, y, w, h = cv.boundingRect(box)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(image)
    plt.title("regions detected by EAST method")
    plt.show()

# returns the bounding box containing all the previously detected bounding boxes
def detect_text_region(image):
    east_boxes = get_EAST_regions(image)
    clone_img = copy.copy(image)

    # Handle empty detection case
    if not east_boxes:
        print("No text regions detected. Using full image.")
        h, w = image.shape[:2]
        return 0, 0, w, h

    draw_detected_bounding_boxes(clone_img, east_boxes)

    points_x = []
    points_y = []
    for box in east_boxes:
        x, y, w, h = cv.boundingRect(box)
        points_x.append(x)
        points_y.append(y)
        points_x.append(x + w)
        points_y.append(y + h)
    pad = 10
    x_min, y_min, x_max, y_max = min(points_x) - pad, min(points_y) - pad, max(points_x) + pad, max(points_y) + pad

    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # show text region in a blue rectangle
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    plt.imshow(image)
    plt.title("a box containing all the detected boxes")
    plt.show()

    return x_min, y_min, x_max, y_max

# Set up DocLing with RapidOCR models
def setup_docling_ocr():
    # Download RapidOCR models from HuggingFace
    print("Downloading RapidOCR models")
    download_path = snapshot_download(repo_id="SWHL/RapidOCR")

    # Setup RapidOcrOptions for English detection
    det_model_path = os.path.join(
        download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
    )

    ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
    )

    pipeline_options = PdfPipelineOptions(
        ocr_options=ocr_options,
    )

    # Create the document converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )

    return converter

# Save image as temporary PDF and use DocLing to process it
def recognize_image_with_docling(image, converter):
    # Save the image as a temporary PDF
    temp_image_path = "temp_image.jpg"
    temp_pdf_path = "temp_image.pdf"

    # Convert RGB to BGR for OpenCV
    image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(temp_image_path, image_bgr)

    # Convert image to PDF (simplified approach)
    from PIL import Image
    img = Image.open(temp_image_path)
    img.save(temp_pdf_path, "PDF", resolution=100.0)

    # Process with DocLing
    conversion_result: ConversionResult = converter.convert(source=temp_pdf_path)
    doc = conversion_result.document

    # Extract text
    text = doc.export_to_markdown()
    lines = text.split('\n')

    # Clean up temporary files
    try:
        os.remove(temp_image_path)
        os.remove(temp_pdf_path)
    except:
        pass

    return lines

# Main function to process an image with EAST + DocLing
def process_image_with_east_docling(image_path):
    # Read the image
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not load image from {image_path}. Check if the file exists.")
        return

    # Convert BGR to RGB for processing
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = image.astype(np.uint8)

    # Setup DocLing OCR with RapidOCR
    converter = setup_docling_ocr()

    # Detect text regions using EAST
    x_min, y_min, x_max, y_max = detect_text_region(image)

    # Crop image to the detected text region
    text_region = image[y_min:y_max, x_min:x_max]
    plt.imshow(text_region)
    plt.title("only text region")
    plt.show()

    # Process the cropped image with DocLing
    lines = recognize_image_with_docling(text_region, converter)

    print("-----------recognized text------------")
    for line in lines:
        if line.strip():  # Only print non-empty lines
            print(line)
    print("--------------------------------------")

    return lines

# Process PDF directly with DocLing (without EAST)
def process_pdf_with_docling(pdf_path):
    # Create output directory if it doesn't exist
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    # Setup DocLing OCR
    print("Setting up DocLing OCR...")

    # Download RapidOCR models from HuggingFace
    print("Downloading RapidOCR models")
    download_path = snapshot_download(repo_id="SWHL/RapidOCR")

    # Setup RapidOcrOptions for English detection
    det_model_path = os.path.join(
        download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
    )

    ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
    )

    pipeline_options = PdfPipelineOptions(
        ocr_options=ocr_options,
    )

    # Convert the document
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )

    print(f"Processing PDF: {pdf_path}")
    conversion_result: ConversionResult = converter.convert(source=pdf_path)
    doc = conversion_result.document
    md = doc.export_to_markdown()

    # Create the output filename based on the input filename
    input_filename = os.path.basename(pdf_path)
    input_filename_without_extension = os.path.splitext(input_filename)[0]
    output_filename = f"{input_filename_without_extension}.md"

    # Write the Markdown output to the file
    output_path = os.path.join(output_directory, output_filename)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(md)

    print(f"Markdown output written to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage for PDF processing
    # pdf_file = "./input/2408.09869v4.pdf"
    # process_pdf_with_docling(pdf_file)

    # Example usage for image processing with EAST + DocLing
    image_file = "datset\9.jpg"
    process_image_with_east_docling(image_file)