import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('/home/ben/Desktop/Space_Engineering/Assignment1/Photos_of_Stars/IMG_3046.jpg', cv2.IMREAD_GRAYSCALE)

# Apply thresholding to separate stars from background
_, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# Find contours of bright spots
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for drawing
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Open a file to save the star information
with open('/home/ben/Desktop/Space_Engineering/Assignment1/stars_data.txt', 'w') as f:
    f.write("Star_Number,X,Y,Radius,Brightness\n")  # header

    # Process each detected star
    for i, contour in enumerate(contours):
        # Calculate center of contour
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Calculate radius based on contour area
            area = cv2.contourArea(contour)
            radius = int(np.sqrt(area / np.pi))
            
            # Calculate brightness
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            brightness = np.mean(image[mask == 255])
            
            # Draw circle at center position
            cv2.circle(result, (cX, cY), max(radius, 10), (0, 255, 0), 2)
            
            # Write star information to file
            f.write(f"{i+1},{cX},{cY},{radius},{brightness:.2f}\n")

# Display result
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Detected Stars with Radius Information')
plt.show()
