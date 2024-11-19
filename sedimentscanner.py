import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

# Function to analyze the uploaded soil image
def analyze_soil_layers(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    original = image.copy()

    # Get the center of the image to focus on the central 10%
    height, width = image.shape[:2]
    center_y, center_x = height // 2, width // 2

    # Define the region of interest (central 10%)
    margin = int(0.1 * height)
    roi = image[center_y - margin:center_y + margin, center_x - margin:center_x + margin]

    # Convert to grayscale for better analysis
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use edge detection to identify contours (soil layers)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours based on the edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list for storing layer information
    layers = []

    total_area = roi.shape[0] * roi.shape[1]  # Total area of the region of interest

    # Process each detected contour
    for idx, contour in enumerate(contours):
        # Skip small contours (noise)
        if cv2.contourArea(contour) < 500:
            continue

        # Create a mask for the current layer
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Calculate the area of the current layer
        layer_area = cv2.contourArea(contour)
        percentage = (layer_area / total_area) * 100

        # Extract the region of interest for this layer
        layer = cv2.bitwise_and(roi, roi, mask=mask)

        # Analyze the grain size and soil type
        grain_size = analyze_grain_size(mask)
        soil_type = classify_soil(layer)

        # Store the layer's results
        layers.append({
            "index": idx + 1,
            "grain_size": grain_size,
            "soil_type": soil_type,
            "area": layer_area,
            "percentage": percentage
        })

        # Annotate the image with the layer information
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]) + (center_x - margin)  # Adjust to original image coordinates
            cY = int(M["m01"] / M["m00"]) + (center_y - margin)

            cv2.putText(image, f"Layer {idx + 1}: {soil_type} ({percentage:.2f}%)", 
                        (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw colored contour and center of the layer
        color = (0, 255, 255)  # Yellow contour for better visibility
        cv2.drawContours(image, [contour], -1, color, 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

    # Save the annotated image
    annotated_image_path = "annotated_soil_image.jpg"
    cv2.imwrite(annotated_image_path, image)

    return layers, annotated_image_path

# Function to analyze grain size based on contours
def analyze_grain_size(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
    if len(areas) == 0:
        return 0
    return sum(areas) / len(areas)

# Function to classify soil type based on image color
def classify_soil(layer):
    # Convert to HSV to analyze color for soil classification
    hsv = cv2.cvtColor(layer, cv2.COLOR_BGR2HSV)
    mean_color = cv2.mean(hsv)[:3]

    # Basic classification based on color in HSV space
    if mean_color[1] < 50 and mean_color[2] > 200:
        return "Sand"
    elif mean_color[2] < 50:
        return "Clay"
    elif 50 < mean_color[1] < 150:
        return "Silt"
    else:
        return "Aggregate"

# Function to open file dialog for selecting an image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            # Analyze the image immediately after uploading
            results, annotated_image_path = analyze_soil_layers(file_path)

            # Display results in the UI
            display_results(results, annotated_image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error processing the image: {e}")

# Function to display results on the UI
def display_results(results, annotated_image_path):
    # Display the annotated image
    img = Image.open(annotated_image_path)
    img = img.resize((400, 400))  # Resize to fit the UI
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk  # Keep a reference to avoid garbage collection

    # Show text results with proper spacing
    result_text = ""
    for result in results:
        result_text += (f"Layer {result['index']}:\n"
                        f"  Soil Type: {result['soil_type']}\n"
                        f"  Grain Size: {result['grain_size']:.2f}\n"
                        f"  Area: {result['area']:.2f} px\n"
                        f"  Percentage of total: {result['percentage']:.2f}%\n\n")
    
    result_label.config(text=result_text)

# Create the main window
root = tk.Tk()
root.title("Soil Analysis Tool")

# Create buttons and labels
upload_button = tk.Button(root, text="Upload Soil Image", command=upload_image)
upload_button.pack(pady=20)

# Create a frame to hold image and result display separately
frame = tk.Frame(root)
frame.pack()

# Label to display the image
img_label = tk.Label(frame)
img_label.grid(row=0, column=0)

# Label to display results
result_label = tk.Label(frame, text="Analysis results will appear here.", justify="left", padx=10)
result_label.grid(row=0, column=1, padx=20)

# Run the Tkinter event loop
root.mainloop()
