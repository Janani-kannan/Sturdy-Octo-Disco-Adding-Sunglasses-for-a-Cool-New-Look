# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!
## Program
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
faceimage=cv2.imread("C:\\Users\\admin\\OneDrive\\Desktop\\DIPT\\janani.jpg")
plt.imshow(faceimage[:,:,::-1]);plt.title("face")
```
<img width="596" height="592" alt="image" src="https://github.com/user-attachments/assets/6bb76a44-8a29-412c-ae85-0865db149485" />

```
glasspng=cv2.imread('C:\\Users\\admin\\OneDrive\\Desktop\\DIPT\\glass.jpeg',-1)
plt.imshow(glasspng[:,:,::-1]);plt.title("GLASSPNG")
```
<img width="842" height="541" alt="image" src="https://github.com/user-attachments/assets/7083bcd0-f549-466e-b497-d09ad0b55b89" />

```
import cv2
import matplotlib.pyplot as plt
glasspng = cv2.imread("C:\\Users\\admin\\OneDrive\\Desktop\\DIPT\\glass.jpeg")
b, g, r = cv2.split(glasspng)
glass_bgr = cv2.merge((b, g, r))
gray = cv2.cvtColor(glasspng, cv2.COLOR_BGR2GRAY)

_, glass_alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

print("BGR shape:", glass_bgr.shape)
print("Alpha shape:", glass_alpha.shape)


plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(glass_bgr, cv2.COLOR_BGR2RGB))
plt.title("Sunglass BGR")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(glass_alpha, cmap="gray")
plt.title("Generated Alpha Mask")
plt.axis("off")

plt.show()
```
<img width="792" height="300" alt="image" src="https://github.com/user-attachments/assets/ed78d378-9964-4b83-8eeb-fd9a40590ab8" />

```
import cv2
import matplotlib.pyplot as plt

# Load face image
face_img = cv2.imread(r"C:\\Users\\admin\\OneDrive\\Desktop\\DIPT\\janani.jpg")
face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# Load sunglasses
glass_bgr = cv2.imread(r"C:\\Users\\admin\\OneDrive\\Desktop\\DIPT\\glass.jpeg")
gray = cv2.cvtColor(glass_bgr, cv2.COLOR_BGR2GRAY)
_, glass_alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Eye detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
eyes = eye_cascade.detectMultiScale(face_gray, 1.2, 5)

# Face dimensions
h, w = face_img.shape[:2]

# Filter eyes (top half of the face)
eyes = [e for e in eyes if e[1] < h // 2]

# Sort by width (largest = real eyes) and pick 2
eyes = sorted(eyes, key=lambda x: -x[2])[:2]

if len(eyes) == 2:
    # Sort left to right
    eyes = sorted(eyes, key=lambda x: x[0])
    x1, y1, w1, h1 = eyes[0]
    x2, y2, w2, h2 = eyes[1]

    # Eye centers
    left_eye = (x1 + w1 // 2, y1 + h1 // 2)
    right_eye = (x2 + w2 // 2, y2 + h2 // 2)

    # Glass size based on eye distance
    eye_distance = int(((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2) ** 0.5)
    glasses_w = int(eye_distance * 2.0)
    glasses_h = int(glass_bgr.shape[0] * (glasses_w / glass_bgr.shape[1]))

    # Resize glasses & mask
    glasses_resized = cv2.resize(glass_bgr, (glasses_w, glasses_h))
    mask_resized = cv2.resize(glass_alpha, (glasses_w, glasses_h))

    # Position (slightly adjusted: up & right)
    center_x = (left_eye[0] + right_eye[0]) // 2
    center_y = (left_eye[1] + right_eye[1]) // 2
    x_offset = center_x - glasses_w // 2 + 20   # move right
    y_offset = center_y - glasses_h // 2 - 27   # move up

    # Ensure within image bounds
    y1, y2 = max(0, y_offset), min(h, y_offset + glasses_h)
    x1, x2 = max(0, x_offset), min(w, x_offset + glasses_w)

    # Crop glasses/mask to fit inside face region
    mask_resized = mask_resized[0:y2 - y1, 0:x2 - x1]
    glasses_resized = glasses_resized[0:y2 - y1, 0:x2 - x1]
    roi = face_img[y1:y2, x1:x2]

    # Masking
    mask_inv = cv2.bitwise_not(mask_resized)
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(glasses_resized, glasses_resized, mask=mask_resized)

    # Combine
    combined = cv2.add(bg, fg)
    face_img[y1:y2, x1:x2] = combined

# Show result
plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Face with Glasses")
plt.show()
```

<img width="525" height="532" alt="image" src="https://github.com/user-attachments/assets/99655946-0fb3-48e2-8519-39798f23f67e" />

```
import matplotlib.pyplot as plt
import cv2

# Show side by side
plt.figure(figsize=(10,5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(faceimage, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Image with glasses
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
plt.title("With Glasses")
plt.axis("off")

plt.show()
```

<img width="1028" height="557" alt="image" src="https://github.com/user-attachments/assets/7efbe278-7d51-4797-a069-e0e3056f6e28" />

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!
