# ================================================
#               **Playing Cards Recognition**
#      *Image Processing and Computer Vision Project*
# ================================================
# 
#       Automation Engineering
#       Academic Year: 2024/2025
# 
# -----------------------------------------------
#                 **Authors**
# -----------------------------------------------
# - Cenni Francesco
# - Mazzaroppi Giulio
# - Pezzolati Giacomo
# ================================================

# ================================================
# 1. IMPORTED LIBRARIES
# ================================================

# Imported libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import re

# Imported libraries needed for benchmark evaluation
import time

# ================================================
# 2. CLASSES DEFINITION
# ================================================

# Blob class
class Blob:
    """
    Represents a rectangular region (blob) detected in an image.
    
    Attributes:
    ---
        x (int): X-coordinate of the top-left corner.
        y (int): Y-coordinate of the top-left corner.
        width (int): Width of the blob.
        height (int): Height of the blob.
        image (np.ndarray): Cropped image of the blob.
    """

    def __init__(self, x: int, y: int, width: int, height: int, image: np.ndarray):
        """
        Initializes a Blob instance.

        Args:
        ---
            x (int): X-coordinate of the top-left corner of the blob.
            y (int): Y-coordinate of the top-left corner of the blob.
            width (int): Width of the blob.
            height (int): Height of the blob.
            image (np.ndarray): Cropped image of the blob.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.image = image


# Card class
class Card:
    """
    Represents a playing card with its image and feature descriptors.
    
    Attributes:
    ---
        image (np.ndarray): Image of the card.
        ID (int): Unique identifier for the card.
        keypoints (list): List of detected keypoints in the card image.
        descriptors (np.ndarray): Feature descriptors associated with keypoints.
    """

    def __init__(self, image: np.ndarray, ID: int, keypoints: list, descriptors: np.ndarray):
        """
        Initializes a Card instance.

        Args:
        ---
            image (np.ndarray): Image of the card.
            ID (int): Unique identifier for the card.
            keypoints (list): List of detected keypoints.
            descriptors (np.ndarray): Feature descriptors associated with the keypoints.
        """
        self.image = image
        self.ID = ID
        self.keypoints = keypoints
        self.descriptors = descriptors

    def draw_keypoints(self) -> np.ndarray:
        """
        Draws keypoints on the card image.

        Returns:
        ---
            np.ndarray: Image with keypoints drawn.
        """
        return cv2.drawKeypoints(self.image, self.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    

# Figure class
class Figure:
    """
    Represent a figure in an image.

    Attributes:
    ---
        area (int): The area of the figure.
        centroid (tuple): The coordinates of the centroid of the figure.
    """

    def __init__(self, area, centroid):
        """
        Initialize a Figure instance.

        Args:
        ---
            area (int): Area of the figure.
            centroid (tuple): Coordinates of the centroid of the figure.
        """
        self.area = area
        self.centroid= centroid
        
    def draw_keypoints(self):
        """
        Draws the keypoints of the figure on the image.

        Returns:
        ---
            np.ndarray: Image with keypoints drawn.
        """
        return cv2.drawKeypoints(self.image, self.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ================================================
# 3. FUNCTIONS DEFINITION
# ================================================

def print_image(img: np.ndarray, colormap: int = cv2.COLOR_BGR2RGB, title: str = None) -> None:
    """
    Displays an image with optional parameters.

    Args:
    ---
        img (np.ndarray): Image to display.
        colormap (int, optional): OpenCV colormap for color conversion. Default is BGR to RGB.
        title (str, optional): Title of the image display. Default is None.
    """
    plt.title(title)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, colormap))
    plt.show()


def compute_background_attributes(img: np.ndarray) -> tuple:
    """
    Computes the mean and inverse covariance matrix of the image's RGB values.

    Args:
    ---
        img (np.ndarray): Input image.

    Returns:
    ---
        tuple: Mean RGB values (np.ndarray) and inverse covariance matrix (np.ndarray).
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    mu = np.mean(img_rgb, axis=0)
    sigma_inv = np.linalg.inv(np.cov(img_rgb.T))
    return mu, sigma_inv


def compute_mahalanobis_distance(img: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> np.ndarray:
    """
    Computes the Mahalanobis distance for each pixel in an image relative to a given distribution.

    Args:
    ---
        img (np.ndarray): Input image.
        mu (np.ndarray): Mean vector of the distribution.
        sigma_inv (np.ndarray): Inverse covariance matrix of the distribution.

    Returns:
    ---
        np.ndarray: 2D array of Mahalanobis distances corresponding to each pixel.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # Vectorization of shape
    img_rgb_flat = img_rgb.reshape(-1, 3)
    diff = img_rgb_flat - mu

    # Efficient matrix multiplication with Einstein's notation (improve fastness)
    distances = np.sqrt(np.einsum('ij,ij->i', diff @ sigma_inv, diff))
    return distances.reshape(h, w) # Reshape distances back to 2D


def segment_image(img: np.ndarray, distances: np.ndarray, threshold: float, kernel_size: int = 0, iterations: int = 0, background: str = 'background', reverse: bool = False) -> np.ndarray:
    """
    1. Segments an image based on a distance map threshold;
    2. Applies dilation and optional erosion to refine the mask;
    3. Supports background handling and reverse thresholding.

    Args:
    ---
        img (np.ndarray): Input image.
        distances (np.ndarray): Distance map for segmentation.
        threshold (float): Threshold to determine foreground pixels.
        kernel_size (int, optional): Size of the kernel for dilation. Default is 0.
        iterations (int, optional): Number of dilation iterations. Default is 0.
        background (str, optional): Type of background handling ('background' or 'white'). Default is 'background'.
        reverse (bool, optional): Whether to reverse the thresholding logic. Default is False.

    Returns:
    ---
        np.ndarray: Processed binary mask of the segmented image.
    """
    if reverse == False:
        binary_mask = (distances >= threshold).astype(np.uint8) * 255
        binary_mask = binary_mask.reshape(img.shape[:2])
    else:
        binary_mask = (distances < threshold).astype(np.uint8) * 255
    
    dilated = cv2.dilate(binary_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
    
    if background == 'background':
        return dilated
    if background == 'erode':
        # To better detect the figure in image for pathological cases: cups with short neck
        return cv2.erode(dilated, np.ones((2, 2), np.uint8), iterations=4)


def detect_blobs(img: np.ndarray, connectivity: int = 8) -> tuple:
    """
    Detects connected components in a binary image using the specified connectivity.

    Args:
    ---
        img (np.ndarray): Input binary image.
        connectivity (int, optional): Pixel connectivity (4 or 8). Default is 8.

    Returns:
    ---
        tuple: Number of labels, labeled image, blob statistics, and blob centroids.
    """
    return cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)


def show_bounding_boxes(img: np.ndarray, blobs: list, bounding_box_colour: tuple = (255, 0, 0) , bounding_box_thickness: int = 4) -> np.ndarray:
    """
    Draws bounding boxes around detected blobs on an image.

    Args:
    ---
        img (np.ndarray): Input image.
        blobs (list): List of Blob objects containing bounding box information.
        bounding_box_colour (tuple, optional): Color of bounding box in BGR format. Default is (255, 0, 0).
        bounding_box_thickness (int, optional): Thickness of rectangle. Default is 4.

    Returns:
    ---
        np.ndarray: New image with bounding boxes drawn around blobs.
    """
    img_blob_boxes = img.copy()
    for blob in blobs:
        startpoint = (blob.x, blob.y)
        endpoint = (startpoint[0] + blob.width, startpoint[1] + blob.height)
        cv2.rectangle(img_blob_boxes, startpoint, endpoint, bounding_box_colour, bounding_box_thickness)
    return img_blob_boxes


def extract_keypoints_and_descriptors(img: np.ndarray, max_key: int, draw_key_points: bool, edge_threshold: int, detector: str) -> tuple:
    """
    Extracts keypoints and descriptors from an image using SIFT or ORB detector.

    Args:
    ---
        img (np.ndarray): Input image.
        max_key (int): Maximum number of keypoints.
        draw_key_points (bool): Whether to draw keypoints on the image.
        edge_threshold (int): Edge threshold for the SIFT detector. The larger the edge_threshold, the less features are filtered out.
        detector (str): Detector type ('sift' or 'orb').

    Returns:
    ---
        tuple: Extracted keypoints and descriptors.
    """

    # Choice of detector and computation of keypoints and descriptors
    if detector == 'sift':
        sift = cv2.SIFT_create(edgeThreshold=edge_threshold, nfeatures=max_key)
        keypoints, descriptors = sift.detectAndCompute(img, None)
    elif detector == 'orb':
        orb = cv2.ORB_create(nfeatures=max_key, scoreType=cv2.ORB_FAST_SCORE)
        keypoints, descriptors = orb.detectAndCompute(img, None)

    # Additionally draw keypoints if specified
    if draw_key_points:
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print_image(img_with_keypoints)

    return keypoints, descriptors


def create_deck(img_bg: np.ndarray,input_path: str, output_path: str, num_cards: int, k_size: int, iter: int) -> bool:
    """
    Creates a deck of segmented card images from the input path and saves them to the output path.

    Args:
    ---
        imb_bg (np.ndarray): Image of the background.
        input_path (str): Directory containing input card images.
        output_path (str): Directory where segmented images will be saved.
        num_cards (int): Expected number of card images to process.
        k_size (int): Kernel size for image dilation during segmentation.
        iter (int): Number of dilation iterations.

    Returns:
    ---
        bool: True if the deck already exists, False if it is created.
    """
    image_files = os.listdir("deck")
    if len(image_files) == num_cards:
        print("Deck already created.")
        return True

    print(f"Error: expected {num_cards} images, found {len(image_files)}.")
    print("Start deck creation...")

    mu, sigma_inv = compute_background_attributes(img_bg)
    threshold = 60
    
    # Process a sequence of images, segment them based on Mahalanobis distance, and save cropped regions of interest.
    for count, i in enumerate(range(1, num_cards + 1), 1):
        # Reads the image. If the image cannot be loaded, it skips to the next
        img = cv2.imread(f"{input_path}{i}.png")

        if img is None:
            continue
        
        # Computes Mahalanobis distances for segmentation using given parameters
        distances = compute_mahalanobis_distance(img, mu, sigma_inv)
        # Segments the image to isolate the foreground or background
        segmented = segment_image(img, distances, threshold, k_size, iter, 'background')
        # Detects connected components and extract blob statistic
        _, _, stats, _ = detect_blobs(segmented)

        # Crops the original image using the bounding box of each detected blob and saves the cropped region to the output path with a sequentially incremented filename
        for x, y, w, h, _ in stats[1:]:
            cropped_img = img[y:y + h, x:x + w]
            cv2.imwrite(f"{output_path}{count}.png", cropped_img)
            count += 1

    print("Deck created!")
    return False


def compute_deck_features(deck_path: str, max_key: int, edge_threshold: int, detector: str, deck_size: int) -> list:
    """
    Extracts keypoints and descriptors from all images in the deck directory and creates a list of Card objects.

    Args:
    ---
        deck_path (str): Directory containing card images.
        max_key (int): Maximum number of keypoints to extract per image.
        edge_threshold (int): Edge threshold used in keypoint detection (SIFT specific).
        detector (str): Keypoint detector type ('sift' or 'orb').
        deck_size (int): Number of cards in the deck.

    Returns:
    ---
        list: List of `Card` objects, each containing the image, ID, keypoints, 
              descriptors for each detected card.
    """

    # Initialize empty list of Card object contained in deck
    deck = np.empty(deck_size, dtype=object)
    for i in range(deck_size):
        deck[i] = []
    
    count = 1
    # Iterates through all image files deck_path
    for img_file in os.listdir(deck_path):
        # Extracts a numeric ID from the filename using regular expressions
        ID = re.search(r'\d+', img_file)
        
        # Detect keypoints and compute descriptors of loaded image
        img = cv2.imread(os.path.join(deck_path, img_file))
        keypoints, descriptors = extract_keypoints_and_descriptors(img, max_key, False, edge_threshold, detector)
        # Displays a progress percentage for feature detection during all process.
        print(f"\rDeck features detection: {int(count / deck_size * 100)}%", end='', flush=True)
    
        count += 1

        # Store Card object with his attributes in deck list if keypoints are successfully detected
        if keypoints != None:
            deck[int(ID.group())-1] = (Card(img, int(ID.group()), keypoints, descriptors))
    
    print("\n")

    return deck


def showFPS(image: np.ndarray, start_time: float = None, text_color: tuple = (255, 255, 255), font_scale: float = 0.8):
    """
    Calculate and display the frames-per-second (FPS) if start_time is provided.

    Args:
    ---
        image: The image frame to display the FPS in.
        start_time (float, optional): The time at which the frame was started, used to calculate the FPS. Defaults to None.
        text_color (tuple, optional): The color of the text (BGR format). Defaults to (255, 255, 255).
        font_scale (float, optional): The scale of the font. Defaults to 0.8.

    Returns:
    ---
        ndarray: The image with the FPS displayed.
    """
    if start_time is not None:
        fps = np.round(1 / (time.time() - start_time), 0)
        fps_text = f"FPS: {fps}"
        cv2.putText(
            image,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            2,
            cv2.LINE_AA
        )

        return image
    

def annotate_image(
    frame: np.ndarray, 
    blob: Blob,
    matched_cards: list,
    bounding_box_colour: tuple = (255, 0, 0), 
    bounding_box_thickness: int = 4, 
    font_scale: float = 0.4, 
    text_color: tuple = (255, 255, 255), 
    text_bg_color: tuple = (0, 0, 0), 
    text_padding: int = 5, 
    line_spacing: int = 20,
    
) -> np.ndarray:
    """
    Draw images bounding boxes and card names.

    This function takes an input image and a blob representing a detected object
    along with a list of matched card names. It draws a bounding box around the 
    blob and annotates it with the card names with customable visual parameters.

    Args:
    ---
        frame (np.ndarray): Input image to be annotated.
        blob (Blob): Blob containing the bounding box information.
        matched_cards (list): List of card names to be annotated on the image.
        bounding_box_colour (tuple): Color of the bounding box in BGR format.
        bounding_box_thickness (int): Thickness of the bounding box lines 
            (default is 4).
        font_scale (float): Scale factor for the text font (default is 0.4).
        text_color (tuple): Color of the text in BGR format (default is white 
            (255, 255, 255)).
        text_bg_color (tuple): Background color for the text in BGR format 
            (default is black (0, 0, 0)).
        text_padding (int): Padding around the text in pixels (default is 5).
        line_spacing (int): Spacing between multiple lines of text (default is 20).

    Returns:
    ---
        np.ndarray: Annotated image with bounding boxes and card names.
    """
    annotated_img = frame.copy()

    # Check if there are any matched cards
    if matched_cards:
        # Draw bounding box
        startpoint = (blob.x, blob.y)
        endpoint = (startpoint[0] + blob.width, startpoint[1] + blob.height)
        cv2.rectangle(annotated_img, startpoint, endpoint, bounding_box_colour, bounding_box_thickness)

        # Annotate blob with card names
        # Put starting point of text over the top left corner of bounding box
        text_x = blob.x
        text_y = blob.y - text_padding

        # If y coord of text exceed the screen limit, the text is put below the bottom left corner
        if text_y < 0:  
            text_y = blob.y + blob.height + text_padding

        # Vertically print card names in the bounding box
        for j, card_label in enumerate(matched_cards):
            text_pos_y = text_y - j * line_spacing
            text_size, _ = cv2.getTextSize(card_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(
                annotated_img,
                (text_x, text_pos_y - text_size[1] - text_padding),
                (text_x + text_size[0] + 2 * text_padding, text_pos_y + text_padding),
                text_bg_color,
                cv2.FILLED
            )
            
            cv2.putText(
                annotated_img,
                card_label,
                (text_x + text_padding, text_pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA
            )

    return annotated_img

# ================================================
# 4. VARIABLES INITIALIZATION
# ================================================

# Dictionary that maps a unique identifier to each card
card_names = {1:"1_CUPS", 2:"2_CUPS", 3:"3_CUPS", 4:"4_CUPS", 5:"5_CUPS", 6:"6_CUPS", 7:"7_CUPS", 8:"KNAVE_CUPS", 9:"KNIGHT_CUPS", 10:"KING_CUPS", 
              11:"1_COINS", 12:"2_COINS", 13:"3_COINS", 14:"4_COINS", 15:"5_COINS", 16:"6_COINS", 17:"7_COINS", 18:"KNAVE_COINS", 19:"KNIGHT_COINS", 20:"KING_COINS",
              21:"1_SWORDS", 22:"2_SWORDS", 23:"3_SWORDS", 24:"4_SWORDS", 25:"5_SWORDS", 26:"6_SWORDS", 27:"7_SWORDS", 28:"KNAVE_SWORDS", 29:"KNIGHT_SWORDS", 30:"KING_SWORDS", 
              31:"1_BATONS", 32:"2_BATONS", 33:"3_BATONS", 34:"4_BATONS", 35:"5_BATONS", 36:"6_BATONS", 37:"7_BATONS", 38:"KNAVE_BATONS", 39:"KNIGHT_BATONS", 40:"KING_BATONS"}  

# ================================================
# 4. PARAMETERS INITIALIZATION
# ================================================

# Note: The following parameters is tuned over a specific setup 
#       configuration for the best performance.

# Video path to try out
video_path = 'test_videos/TestVideo4.mp4'

# Video output path
output_video = "video_results/out.mp4" 

# Segmentation parameters
segment_threshold = 20
kernel_segmentation = 3
iterations_segmentation = 5

# Keypoint and detector parameters
max_key_deck = 2000
max_key_target = 3000
edge_threshold = 100
detector = 'sift'

# Matching method either Flann or BruteForce
matcher = 'bf'

if matcher == 'bf':
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
elif matcher == 'flann':
    index_params = dict(algorithm=1, trees=5) # For KDTree
    search_params = dict(checks=50) # Number of checks
    flann = cv2.FlannBasedMatcher(index_params, search_params)

# Minimum number of matches required for robust ransac homography
MIN_MATCH_COUNT = 4

# Lowe's ratio parameters
lowe_ratio1 = 0.8
lowe_ratio2 = 0.8

# Choosing logic parameters
confidence = 0.3
abs_thresh1 = 8
abs_thresh2 = 20

# Card parameters
images_path = "images"          # Path of train dataset
deck_path = "deck"              # Path of filtered dataset
cards_number = 40               # Cards number in the dataset

# Background image for deck creation
img_bg = cv2.imread("backgrounds/Background_deck.jpeg")

# Blob "band pass" filter    
min_width = 60
min_height = 60
max_width = 300
max_height = 300

# ================================================
# 5. DECK CREATION
# ================================================

# Create the deck of the card in images_path
create_deck(img_bg, images_path + "/Card", deck_path + "/Card", cards_number, kernel_segmentation, iterations_segmentation)

# Compute deck feature for each card in the deck_path
deck = compute_deck_features(deck_path, max_key_deck, edge_threshold, detector, cards_number)

# Confirm the deck feature creation
print("Deck features created\n")

# ================================================
# 6. DYNAMIC BACKGROUND TAKING
# ================================================

### Background taking initialization GUI
background_square = 0.5             # Relative dimension of frame size
font = cv2.FONT_HERSHEY_SIMPLEX     # Font of text in the rectangle for background taking
font_scale = 5                      # Scale of timer in rectangle for background taking
thickness = 10                      # Thickness of rectangle for background 

print("Prepare to take background photo\n")

# Load the video and create a VideoCaputre instance
cap = cv2.VideoCapture(video_path)

# Timer for photo taking
duration = 2
start_time = time.time()
time_passed = 0

# FPS Saturation 
max_fps = 20
max_frame_delay = 1.0 / max_fps

if cap.isOpened():
    while time_passed <= duration:

        # Count time passed from opening
        time1 = time.time()

        # Count time passed from opening
        time_passed = time.time() - start_time 

        # Capture frame from the camera
        ret, frame = cap.read()
        
        # Break if image isn't correctly acquired
        if not ret:
            # Release the Video if ret is false
            print("Error: Could not read frame.")
            break

        # Break if 'q' key is pressed -> meaning "quit"
        if cv2.waitKey(1) == ord('q'):
            break
        
        # Frame cropping
        h, w, _ = frame.shape
        dh = h * background_square
        dw = w * background_square
        
        frame_text = frame.copy()
        
        # Get the timer text dimension 
        text =  str(int(duration - time_passed))
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size
        
        # Text centering
        x = (w - text_width) // 2
        y = (h + text_height) // 2
        
        # Print the timer and frame rectangle cropped in the frame
        cv2.rectangle(frame_text, (int((w - dw) / 2), int((h - dh) / 2)), (int((w + dw) / 2), int((h + dh) / 2)), (0, 0, 255), 4)
        cv2.putText(frame_text, text, (x, y), font, font_scale, (0, 0, 255), thickness)

        # Show the background acquisition screen
        cv2.imshow('Background_acquisition', frame_text)

        ### FPS Saturation
        # Record the time after the operations above
        time2 = time.time()
        # If the frame took less than max_frame_delay time to execute, wait for the remaining time
        if time2 - time1 < max_frame_delay:
            time.sleep(max_frame_delay - (time2 - time1))

# Close all OpenCV windows
cv2.destroyAllWindows()

# Save and show background acquired
background = frame[int((h - dh) / 2):int((h + dh) / 2), int((w - dw) / 2):int((w + dw) / 2)]
cv2.imwrite("backgrounds/Background_captured.png", background)
cv2.imshow('Background', background)
    
# Wait for 1000 milliseconds (1 seconds) and close the window automatically
cv2.waitKey(1000)  # Time in milliseconds
cv2.destroyAllWindows()

print("Background correctly taken, starting recognition\n")

# ================================================
# 7. VIDEO STREAM APPLICATION
# ================================================

# Initialize the VideoWriter to save the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fps = max_fps  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

# VideoWriter object creation
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Number of blobs in a frame
num_blobs = 0

while cap.isOpened():
    # Start time for FPS calculation
    start_time = time.time()

    # Capture frame from the camera
    ret, frame = cap.read()

    # Break if image isn't correctly acquired
    if not ret or frame is None:
        # Release the Video if ret is false
        cap.release()
        print("Released Video Resource")
        break
    
    # Break if 'q' key is pressed -> meaning "quit"
    if cv2.waitKey(1) == ord('q'):
        break

    # Image segmentation of foreground objects given mahalanobis distance computed over dynamically acquired background
    img_try = frame
    mu_bg, sigma_inv_bg = compute_background_attributes(background)
    d_M = compute_mahalanobis_distance(img_try, mu_bg, sigma_inv_bg)
    img_seg = segment_image(img_try, d_M, segment_threshold, 2, 4, 'background')
    
    # Filter out background pixels, keeping only the foreground
    img_try = cv2.bitwise_and(img_try, img_try, mask=img_seg)

    # Save filtered out blob detected in a list of Blob object
    numlabels, img_blob, stats, centroids = detect_blobs(img_seg)
    blobs = [
        Blob(x, y, w, h, img_try[y:y + h, x:x + w])
        for x, y, w, h, _ in stats[1:]
        if w >= min_width and h >= min_height and w <= max_width and h <= max_height
    ]

    # Reinitialize array of matched cards when blobs number change in image
    if num_blobs != len(blobs):
        matched_cards = np.empty(len(blobs), dtype=object)
        
        # Empty list for the card names a blob may contain
        for i in range(len(blobs)):
            matched_cards[i] = []  

    # For each blob in the image
    for index_blob, blob in enumerate(blobs):
        # Extract keypoints and descriptors
        keypoints, descriptors = extract_keypoints_and_descriptors(blob.image, max_key_target, False, edge_threshold, detector)

        # Initialize accumulator array for counting robust ransac homography card matches
        index_count = np.zeros(len(deck), dtype=int)

        # Initialize unidirectional card match accumulator, needed for seed detection
        index_count2 = np.zeros(len(deck), dtype=int)

        # If descriptors are found
        if descriptors is not None and len(descriptors) > 1:
            # For every card in the deck, perform feature matching with the current blob's descriptors
            for i, card in enumerate(deck):      
                if card != []:
                    # Perform descriptor matching between [Blob (Test Image) - Card (Train Image)] using the specified matcher
                    if matcher == 'bf':
                        matches = bf.knnMatch(descriptors, card.descriptors, k=2)
                    elif matcher == 'flann':
                        matches = flann.knnMatch(descriptors, card.descriptors, k=2)
                    
                    # Initialize an empty list to store good matches after validation
                    good = []

                    # Lowe's ratio test for filtering out ambiguous or weak matches
                    for m, n in matches:
                        if m.distance < lowe_ratio1 * n.distance:
                            # Increment unidirectional card match accumulator
                            index_count2[i] += 1

                            # Cross-check validation with reverse matching: [Card (Train Image) - Blob (Test Image)] using the specified matcher
                            if matcher == 'bf':
                                reverse_matches = bf.knnMatch(card.descriptors[m.trainIdx].reshape(1, -1), descriptors, k=2)
                            elif matcher == 'flann':
                                reverse_matches = flann.knnMatch(card.descriptors[m.trainIdx].reshape(1, -1), descriptors, k=2)
                            
                            # Validate reverse match with Lowe's ratio and ensure index consistency
                            for rm in reverse_matches: 
                                #  Ensures that knnMatch returned only the two nearest neighbors
                                if len(rm) == 2:
                                    p, q = rm 
                                    if p.distance < lowe_ratio2 * q.distance and rm[0].trainIdx == m.queryIdx: 
                                        good.append(m)

                    if len(good) > MIN_MATCH_COUNT:
                        
                        # building the correspondences arrays of good matches
                        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                        dst_pts = np.float32([card.keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                        
                        # Using RANSAC to estimate a robust homography. 
                        # It returns the homography M and a mask for the discarded points
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        # Mask of discarded point used in visualization (1 (inlier) - 0 (outlier))
                        matchesMask = np.array(mask.ravel().tolist())

                        # Save the number of inliers for the card at index i
                        index_count[i] = len(matchesMask[matchesMask==1])
            
        ### Heuristics for pathological cards: choice of the seed
        # Find the seed of the largest cluster of card matches
        index_count2[index_count2 < abs_thresh2] = 0
        if np.max(index_count2) > 0:
            seed = np.argmax(np.sum(index_count2.reshape((4, -1)), axis=1), )
        
        # Normalization of index count by the highest count to compute relative confidence
        max_count = max(index_count)
        index_count_rel1 = index_count
        if max_count > 0:
            index_count_rel1 = index_count / max_count
        
        # Label for pathological cards set to None (no pathological case detected)
        extracted_figure_ID = None

        ### Heuristics for pathological cards: we segment inside the card, knowing the seed and counting the blobs we'll know the card
        # "n of seed"
        if np.max(index_count) < abs_thresh1 and np.max(index_count2) > abs_thresh2: # Discriminate cards with repetitive patterns

            # Segmentation of the card patterns
            mu_bg, sigma_inv_bg = compute_background_attributes(background)
            d_M = compute_mahalanobis_distance(blob.image, mu_bg, sigma_inv_bg)

            # Gaussian blur for robustness to noise and flickering (only take larger feature)
            # Performs dilation followed by erosion (Closing) to fill small holes and connect nearby objects in binary images
            img_seg = segment_image(cv2.GaussianBlur(blob.image, (31, 31), 10), d_M, 95, 3, 3, 'erode') 
            
            # Blobs found in the segmented card
            numlabels, img_blob, stats, centroids = detect_blobs(img_seg)

            # List of Figure inside the card
            figures = []
            
            # If there are more than one blob
            if numlabels > 1:
                # Store in figures only the foreground figure with area > 0
                for index in range(len(stats)):                       
                    if index > 0:
                        figures.append(Figure(stats[index, 4], centroids[index]))
            
            # Assign the figure ID by counting the figures stored and the detected seed
            if seed == 0:
                extracted_figure_ID = np.floor(len(figures) / 2)        # Cups are first ten cards
            if seed == 1:
                extracted_figure_ID = 10 + len(figures)                 # Coins are from eleven to twenty
                        
        # Normalization of index count2 by the highest count2 to compute relative confidence
        max_count = max(index_count2)
        index_count_rel2 = index_count2
        if max_count > 0:
            index_count_rel2 = index_count2 / max_count
        
        # Add card names to matched_cards for specific blob
        for i in range(len(index_count)):
            # If the relative confidence and match count is above their threshold
            if index_count_rel1[i] >= confidence and index_count[i] > abs_thresh1:
                # Retrieve card label using the id card in the deck from dictionary
                card_label = f"{card_names[deck[i].ID]}"
                # Hold Mechanism: add card label if not already present in matched_cards list
                if card_label not in matched_cards[index_blob]:
                    matched_cards[index_blob].append(card_label)

        # Add card names to matched_cards for specific blob in pathological case
        if extracted_figure_ID != None:
            if extracted_figure_ID in card_names:
                # Retrieve card label using the id card in the deck from dictionary
                card_label = f"{card_names[extracted_figure_ID]}"
                # Hold Mechanism: add card label if not already present in matched_cards list
                if card_label not in matched_cards[index_blob]:
                    matched_cards[index_blob].append(card_label)
            else:
                # Error in case counted figure = 0
                print(f"Error: ID {extracted_figure_ID} not found in card_names")

        frame = annotate_image(frame, blob, matched_cards[index_blob])

    # FPS saturation
    # Record the time after the operations above
    time2 = time.time()
    # If the frame took less than max_frame_delay time to execute, wait for the remaining time
    if time2 - start_time < max_frame_delay:
        time.sleep(max_frame_delay - (time2 - start_time))

    # Display the current view with annotated blobs and FPS
    frame = showFPS(frame, start_time)
    cv2.imshow('Game frame', frame)

    # Update the number of blobs in the frame
    num_blobs = len(blobs)

    # Write the current frame to the video file
    video_writer.write(frame)

# Release the VideoWriter object when finished
video_writer.release()

# Release camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()