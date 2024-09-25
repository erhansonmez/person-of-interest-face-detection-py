import os
import cv2
import face_recognition
import numpy as np

# Dictionary defined with keys and colors
colorArr = {
    "admin": (56, 218, 255),
    "primary_asset": (211, 0, 148),
    "asset": (34, 139, 34),
    "threat": (0, 0, 255),
}

# Checking folder structures and automatically creating person_data
base_folders = {
    "admins": "admin",
    "primary_assets": "primary_asset",
    "assets": "asset",
    "threats": "threat"
}

# Function to automatically create person_data
def generate_person_data(base_folders):
    person_data = []
    for folder, role in base_folders.items():
        # Traverse all files in the folder
        for filename in os.listdir(folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):  # Only consider image files
                # Create the file path
                filepath = os.path.join(folder, filename)
                # Name is taken from the file name (without extension)
                name = os.path.splitext(filename)[0].capitalize()
                # Add role, name, and file path
                person_data.append((role, name, filepath))
    return person_data

# Automatically generating person_data
person_data = generate_person_data(base_folders)

# Merging face encodings and names
face_image_paths = [(path, name, colorArr[role]) for role, name, path in person_data]

face_encodings = []
face_names = []
face_colors = []

# Add encodings for all faces
for image_path, name, color in face_image_paths:
    try:
        image = face_recognition.load_image_file(image_path)
        # Face recognition may not work on non-human faces, so we use a try-except block
        encodings = face_recognition.face_encodings(image)
        if encodings:
            face_encodings.append(encodings[0])
            face_names.append(name)
            face_colors.append(color)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def draw_machine_symbol(img, top_left, bottom_right, color, thickness=1, dash_length=6, radius=10, opacity=0.5):
    x1, y1 = top_left
    x2, y2 = bottom_right

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Calculate the size of the square (height and width)
    width = x2 - x1
    height = y2 - y1

    rate = width / 400

    # If the square shrinks, adjust the parameters
    thickness = max(thickness, int(6 * rate))
    dash_length = max(dash_length, int(20 * rate))
    radius = max(radius, int(15 * rate))

    cv2.line(img, (center_x, y1), (center_x, y1 + 10), color, int(thickness*2))
    cv2.line(img, (center_x, y2), (center_x, y2 - 10), color, int(thickness*2))
    cv2.line(img, (x1, center_y), (x1 + 10, center_y), color, int(thickness*2))
    cv2.line(img, (x2, center_y), (x2 - 10, center_y), color, int(thickness*2))

    # Draw the four corner ellipses directly on the main image at 100% opacity
    cv2.ellipse(img, (x1 + radius*2, y1 + radius*2), (radius*2, radius*2), 180, 0, 90, color, int(thickness*2))  # Top left corner
    cv2.ellipse(img, (x2 - radius*2, y1 + radius*2), (radius*2, radius*2), 270, 0, 90, color, int(thickness*2))  # Top right corner
    cv2.ellipse(img, (x2 - radius*2, y2 - radius*2), (radius*2, radius*2), 0, 0, 90, color, int(thickness*2))    # Bottom right corner
    cv2.ellipse(img, (x1 + radius*2, y2 - radius*2), (radius*2, radius*2), 90, 0, 90, color, int(thickness*2))   # Bottom left corner

    # Create an overlay layer to apply transparency for the edges
    overlay = img.copy()

    # Draw the dashed lines for the edges (top, bottom, left, right edges)
    for i in range(x1 + radius, x2 - radius, dash_length * 2):
        cv2.line(overlay, (i, y1), (min(i + dash_length, x2 - radius), y1), color, thickness)

    for i in range(x1 + radius, x2 - radius, dash_length * 2):
        cv2.line(overlay, (i, y2), (min(i + dash_length, x2 - radius), y2), color, thickness)

    for i in range(y1 + radius, y2 - radius, dash_length * 2):
        cv2.line(overlay, (x1, i), (x1, min(i + dash_length, y2 - radius)), color, thickness)

    for i in range(y1 + radius, y2 - radius, dash_length * 2):
        cv2.line(overlay, (x2, i), (x2, min(i + dash_length, y2 - radius)), color, thickness)

    # Mix the original image with the overlay to add transparency
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 5, img)

def draw_samaritan_symbol(img, top_left, bottom_right, color_triangle, color_circle, thickness=2):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Calculate the radius (distance between two points)
    radius = (x2 - x1) // 2

    # Adjust the size of the triangle
    triangle_size = radius * 1.2  # Side length of the triangle

    # Determine the corners of the inverted triangle
    x, y = (x1 + x2) // 2, (y1 + y2) // 2  # Center point
    triangle_points = np.array([
        [x, y + int(triangle_size // 2)],  # Bottom point
        [x - int(triangle_size // 2), y - int(triangle_size // 2)],  # Top left point
        [x + int(triangle_size // 2), y - int(triangle_size // 2)]  # Top right point
    ])

    # Draw the inverted triangle (red)
    cv2.polylines(img, [triangle_points], isClosed=True, color=color_triangle, thickness=thickness)

    # Draw the circles (gray)
    cv2.circle(img, (x, y), radius, color_circle, thickness=thickness)
    cv2.circle(img, (x, y), int(radius * 0.9), color_circle, thickness=thickness)

    # Small gray lines (on the sides)
    line_length = int(radius * 0.3)
    gap = int(radius * 0.15)

    # Left line
    cv2.line(img, (x - radius + gap, y), (x - radius + gap + line_length, y), color_circle, thickness=thickness)
    # Right line
    cv2.line(img, (x + radius - gap, y), (x + radius - gap - line_length, y), color_circle, thickness=thickness)

    # Small circle in the center (gray)
    cv2.circle(img, (x, y), int(radius * 0.05), color_circle, thickness=thickness)

def draw_text_with_background(img, text, position, font, font_scale, text_color, background_color, opacity=0.5, padding=5, thickness=1):
    # Get the text dimensions
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the background rectangle for the text
    x, y = position
    top_left = (x, y - text_height - padding)
    bottom_right = (x + text_width + padding * 2, y + baseline + padding)

    # Create a copy of the original image as an overlay
    overlay = img.copy()

    # Draw the background on the overlay (rectangle)
    cv2.rectangle(overlay, top_left, bottom_right, background_color, cv2.FILLED)

    # Mix the overlay with the original image for transparency
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # Draw the text
    cv2.putText(img, text, (x + padding, y), font, font_scale, text_color, thickness)

# Capture video from the camera
video_capture = cv2.VideoCapture(0)

# Track the frame count to perform face recognition periodically
frame_count = 0
face_locations = []
face_encodings_in_frame = []

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Cannot access the camera")
        break

    # Perform face recognition every 5 frames to improve performance
    frame_count += 1
    if frame_count % 5 == 0:
        # Reduce resolution and convert to RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the smaller frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Instead of updating faces on every frame, display faces detected in the previous frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_in_frame):
        name = "UNKNOWN"
        borderColor = (100, 100, 100)

        # Compare with all face encodings
        matches = face_recognition.compare_faces(face_encodings, face_encoding)

        if True in matches:
            first_match_index = matches.index(True)
            name = face_names[first_match_index]
            borderColor = face_colors[first_match_index]

        # Restore to original size (since the frame was processed in a smaller size)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rounded dashed rectangle
        #draw_machine_symbol(frame, (left, top), (right, bottom), borderColor, thickness=2, dash_length=6, radius=10, opacity=0.4)
        draw_samaritan_symbol(frame, (left, top), (right, bottom), (0,0,255), (180, 180, 180), thickness=2)  # 200-size red inverted triangle

        width = top - left
        rate = width / 400

        # Show the name on the screen
        vertical_center = top + (bottom - top) // 2
        draw_text_with_background(frame, name, (right + 10, vertical_center), cv2.FONT_HERSHEY_DUPLEX, 0.52, text_color=borderColor, background_color=borderColor, opacity=0.1, padding=5, thickness=1)

    # Show the result image
    cv2.imshow('Video', frame)

    # Wait for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
video_capture.release()
cv2.destroyAllWindows()
