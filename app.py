import os
import pygame
import sys
from pygame.locals import *
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Disable oneDNN custom operations in TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BOUNDARY = 5

# Image saving settings
IMGSAVE = False
img_count = 1

# Load the trained model
try:
    model = load_model("bestmodel_new.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Label dictionary for the digit classes
labels = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
          6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize Pygame
pygame.init()

FONT = pygame.font.Font(None, 18)
DISPLAYSURF = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Digit Recognition Board")

# Initialize variables
is_writing = False
num_xcord = []
num_ycord = []

# Define button area and text
clear_button_rect = pygame.Rect(550, 420, 80, 40)  # x, y, width, height
clear_button_text = FONT.render("Clear", True, BLACK, WHITE)
clear_button_text_rect = clear_button_text.get_rect(center=clear_button_rect.center)

def preprocess_image(img_arr):
    """Preprocess the drawn image to match model input."""
    # Resize the image to 28x28
    image = cv2.resize(img_arr, (28, 28))

    # Check if the image already has 1 channel (grayscale)
    if len(image.shape) == 3:  # If the image has 3 channels (RGB/BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Normalize the pixel values to [0, 1] range
    image = image / 255.0

    # Reshape the image to match the model input shape (28x28x1)
    image = image.reshape(1, 28, 28, 1)

    return image


def draw_bounding_box_and_label(rect_min_x, rect_max_x, rect_min_y, rect_max_y, label):
    """Draw bounding box and label on the screen."""
    text_surface = FONT.render(label, True, RED, WHITE)
    text_rect = text_surface.get_rect()
    text_rect.center = (rect_min_x + (rect_max_x - rect_min_x) // 2, rect_min_y - 10)

    # Draw bounding box and label
    pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)
    DISPLAYSURF.blit(text_surface, text_rect)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and is_writing:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            num_xcord.append(xcord)
            num_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            is_writing = True
            num_xcord.clear()  # Clear coordinates on new drawing
            num_ycord.clear()

        if event.type == MOUSEBUTTONUP:
            is_writing = False

            if num_xcord and num_ycord:  # Ensure lists are not empty
                num_xcord = sorted(num_xcord)
                num_ycord = sorted(num_ycord)

                # Get bounding box of drawn digit
                rect_min_x, rect_max_x = max(num_xcord[0] - BOUNDARY, 0), min(640, num_xcord[-1] + BOUNDARY)
                rect_min_y, rect_max_y = max(num_ycord[0] - BOUNDARY, 0), min(num_ycord[-1] + BOUNDARY, 480)

                # Ensure the drawn area is of significant size to make a prediction
                if rect_max_x - rect_min_x > 20 and rect_max_y - rect_min_y > 20:
                    # Create image array of the drawn digit
                    img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,
                              rect_min_y:rect_max_y].T.astype(float)

                    if IMGSAVE:
                        cv2.imwrite(f"image_{img_count}.png", img_arr)
                        img_count += 1

                    # Preprocess the image to match the model input requirements
                    image = preprocess_image(img_arr)

                    # Make a prediction
                    prediction = np.argmax(model.predict(image))
                    label = str(labels[prediction])

                    # Debugging the model's output
                    print(f"Predicted label: {label}")  # Check the predicted label
                    # Draw bounding box and prediction label
                    draw_bounding_box_and_label(rect_min_x, rect_max_x, rect_min_y, rect_max_y, label)

                # Clear coordinates after prediction is made
                num_xcord.clear()
                num_ycord.clear()

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

        # Check if mouse clicked on the "Clear" button
        if event.type == MOUSEBUTTONDOWN:
            if clear_button_rect.collidepoint(event.pos):  # Check if click is inside the button
                DISPLAYSURF.fill(BLACK)  # Clear the screen

    # Draw the "Clear" button
    pygame.draw.rect(DISPLAYSURF, WHITE, clear_button_rect)
    DISPLAYSURF.blit(clear_button_text, clear_button_text_rect)

    pygame.display.update()
