from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
import requests

# Initialize Flask app
app = Flask(__name__)

# Set paths for input and output directories
inputPath = "E:/projects/uyirhackathon/AITrafficManagementSystem/test_images/"
outputPath = "E:/projects/uyirhackathon/AITrafficManagementSystem/output_images/"

# URLs for YOLOv3 weights and config files
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
YOLO_CFG_URL = "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true"
YOLO_CLASSES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# File paths
yolo_weights_file = os.path.join(os.getcwd(), "yolov3.weights")
yolo_cfg_file = os.path.join(os.getcwd(), "yolov3.cfg")
yolo_classes_file = os.path.join(os.getcwd(), "coco.names")

# Function to download files from URLs
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists. Skipping download.")

# Download YOLOv3 files if not already present
download_file(YOLO_WEIGHTS_URL, yolo_weights_file)
download_file(YOLO_CFG_URL, yolo_cfg_file)
download_file(YOLO_CLASSES_URL, yolo_classes_file)

# Load YOLO
net = cv2.dnn.readNet(yolo_weights_file, yolo_cfg_file)

# Read class labels
with open(yolo_classes_file, 'r') as f:
    classes = f.read().strip().split('\n')

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Vehicle detection function
def detectVehicles(filename):
    global net, inputPath, outputPath
    img = cv2.imread(inputPath + filename)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = classes[class_id]  # Label from coco.names
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save output image
    outputFilename = outputPath + "output_" + filename
    cv2.imwrite(outputFilename, img)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image uploads and detection
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save uploaded file to the input directory
    filename = file.filename
    input_file_path = os.path.join(inputPath, filename)
    file.save(input_file_path)

    # Perform vehicle detection
    detectVehicles(filename)

    # Return the output image
    output_filename = "output_" + filename
    return send_from_directory(outputPath, output_filename)

# Route for accessing the uploaded image directory (optional)
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(inputPath, filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)