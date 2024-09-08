import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pygame
from sklearn.cluster import KMeans

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("KMeans Image Processing")

# Color
BACKGROUND = (214, 214, 214)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Font
font = pygame.font.Font(None, 36)

# Load image
uploaded_image = None
uploaded_image_path = None
processed_image = None
processed_image_path = "processed_image.png"

running = True
FPS = 60
clock = pygame.time.Clock()

K = 5  # Number of clusters


def load_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if file_path:
        return pygame.image.load(file_path), file_path

    return None, None


while running:
    clock.tick(FPS)
    screen.fill(BACKGROUND)

    # Draw button load image
    pygame.draw.rect(screen, BLACK, (300, 50, 200, 50))
    load_image_text = font.render("Load Image", True, WHITE)
    screen.blit(load_image_text, (330, 60))

    # Draw button + and - and number of clusters
    pygame.draw.rect(screen, BLACK, (300, 150, 50, 50))
    pygame.draw.rect(screen, BLACK, (450, 150, 50, 50))
    plus_text = font.render("+", True, WHITE)
    minus_text = font.render("-", True, WHITE)
    screen.blit(plus_text, (320, 160))
    screen.blit(minus_text, (470, 160))
    num_clusters = font.render(str(K), True, BLACK)
    screen.blit(num_clusters, (395, 160))

    # Draw two rectangles for images
    rect1_x, rect1_y, rect_width, rect_height = 50, 230, 300, 300
    rect2_x, rect2_y = 450, 230  # Second rectangle

    pygame.draw.rect(screen, BLACK, (rect1_x, rect1_y, rect_width, rect_height), 5)
    pygame.draw.rect(screen, BLACK, (rect2_x, rect2_y, rect_width, rect_height), 5)

    # Draw uploaded image in the first rectangle
    if uploaded_image:
        image = pygame.transform.scale(uploaded_image, (rect_width, rect_height))
        screen.blit(image, (rect1_x, rect1_y))

        # Get image file size
        file_size = os.path.getsize(uploaded_image_path)
        file_size_kb = file_size / 1024
        file_size_text = f"{file_size_kb:.1f} KB"

        # Display image size below the first rectangle
        image_size_text = font.render(file_size_text, True, BLACK)
        screen.blit(image_size_text, (rect1_x, rect1_y + rect_height + 10))

    # Draw processed image in the second rectangle (if available)
    if processed_image:
        image = pygame.transform.scale(processed_image, (rect_width, rect_height))
        screen.blit(image, (rect2_x, rect2_y))

        # Save processed image to file to get its size
        pygame.image.save(processed_image, processed_image_path)
        processed_file_size = os.path.getsize(processed_image_path)
        processed_file_size_kb = processed_file_size / 1024
        processed_file_size_text = f"{processed_file_size_kb:.1f} KB"

        # Display processed image size below the second rectangle
        processed_image_size_text = font.render(processed_file_size_text, True, BLACK)
        screen.blit(processed_image_size_text, (rect2_x, rect2_y + rect_height + 10))

    if uploaded_image:
        # Display image in the first rectangle
        uploaded_image = pygame.transform.scale(
            uploaded_image, (rect_width, rect_height)
        )
        screen.blit(uploaded_image, (rect1_x, rect1_y))

        # Convert image to numpy array
        image = pygame.surfarray.array3d(uploaded_image)
        h, w, _ = image.shape
        image = image.reshape((h * w, 3))
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=K, random_state=0).fit(image)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        processed_image = np.zeros_like(image)
        for i in range(len(image)):
            processed_image[i] = centers[labels[i]]

        # Convert processed image to surface and display in the second rectangle
        processed_image = processed_image.reshape((h, w, 3))
        processed_image = pygame.surfarray.make_surface(processed_image)
        screen.blit(processed_image, (rect2_x, rect2_y))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()

            # Load image
            if 300 <= x <= 500 and 50 <= y <= 100:
                uploaded_image, uploaded_image_path = load_image()

            # Button +
            if 300 <= x <= 350 and 150 <= y <= 200:
                K += 1

            # Button -
            if 450 <= x <= 500 and 150 <= y <= 200:
                K = max(1, K - 1)

    pygame.display.flip()

pygame.quit()

# Remove the processed image file after quitting
if os.path.exists(processed_image_path):
    os.remove(processed_image_path)