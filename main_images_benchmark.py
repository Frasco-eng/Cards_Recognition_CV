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
import pandas as pd

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



# ================================================
# 4. VARIABLES INITIALIZATION
# ================================================

# Dictionary that maps a unique identifier to each card
card_names = {1:"1_CUPS", 2:"2_CUPS", 3:"3_CUPS", 4:"4_CUPS", 5:"5_CUPS", 6:"6_CUPS", 7:"7_CUPS", 8:"KNAVE_CUPS", 9:"KNIGHT_CUPS", 10:"KING_CUPS", 
              11:"1_COINS", 12:"2_COINS", 13:"3_COINS", 14:"4_COINS", 15:"5_COINS", 16:"6_COINS", 17:"7_COINS", 18:"KNAVE_COINS", 19:"KNIGHT_COINS", 20:"KING_COINS",
              21:"1_SWORDS", 22:"2_SWORDS", 23:"3_SWORDS", 24:"4_SWORDS", 25:"5_SWORDS", 26:"6_SWORDS", 27:"7_SWORDS", 28:"KNAVE_SWORDS", 29:"KNIGHT_SWORDS", 30:"KING_SWORDS", 
              31:"1_BATONS", 32:"2_BATONS", 33:"3_BATONS", 34:"4_BATONS", 35:"5_BATONS", 36:"6_BATONS", 37:"7_BATONS", 38:"KNAVE_BATONS", 39:"KNIGHT_BATONS", 40:"KING_BATONS"}  

# ================================================
# 5. PARAMETERS INITIALIZATION
# ================================================

# Note: The following parameters is tuned over a specific setup 
#       configuration for the best performance.

# Segmentation parameters
segment_threshold = 50
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
confidence = 0.1
abs_thresh1 = 10

# Card parameters
images_path = "images"          # Path of train dataset
deck_path = "deck"              # Path of filtered dataset
cards_number = 40               # Cards number in the dataset

# Background image for deck creation
img_bg = cv2.imread("backgrounds/Background_deck.jpeg")

# Blob "band pass" filter    
min_width = 60
min_height = 60
max_width = 10000
max_height = 10000

# dictionaries for result saving in result.xlsx
parameters = {}
results = {}


label = ['Run1']  # This will be the label for the current run


# ================================================
# 6. DECK CREATION
# ================================================

# Create the deck of the card in images_path
create_deck(img_bg, images_path + "/Card", deck_path + "/Card", cards_number, kernel_segmentation, iterations_segmentation)

# Compute deck feature for each card in the deck_path
deck = compute_deck_features(deck_path, max_key_deck, edge_threshold, detector, cards_number)

# Confirm the deck feature creation
print("Deck features created\n")

# ================================================
# 7. BENCHMARK PARAMETERS INITIALIZATION
# ================================================

# parameters for benchmark initializations
numtest = 14
elapsed_times = []
found_cards = 0
found_true_cards = 0
not_found_cards=0
right_confidences = 0
not_right_confidences = 0
min_right_confidence = 100
max_wrong_confidence = 0

# True cards that the algorithm should find in every test
Test_True_Cards =   [['7_SWORDS'],
                    ['2_COINS', '1_CUPS'],
                    ['1_CUPS'],
                    ['2_BATONS', 'KNAVE_BATONS'],
                    ['7_SWORDS', 'KNIGHT_SWORDS'],
                    ['7_SWORDS', '7_BATONS'],
                    ['7_BATONS', '7_SWORDS'],
                    [],
                    ['7_BATONS', '5_SWORDS', '2_SWORDS'], 
                    ['KING_COINS'], 
                    ['KNAVE_BATONS', '7_BATONS', '7_SWORDS', '5_SWORDS', '2_SWORDS'], 
                    ['KING_COINS'],
                    ['7_BATONS', '5_SWORDS', '2_SWORDS'],
                    ['7_BATONS', '5_SWORDS', '2_SWORDS']]

# check if there are no typos 
for cards in Test_True_Cards:
    for card in cards:
        if card not in card_names.values():
            print(f"{card} not found in dictionary")
            exit()

#count the number of total cards in the tests to compute benchmarks
total_cards = sum(len(cards) for cards in Test_True_Cards)
# ================================================
# 8. Recognition algorithm applied to every test image
# ================================================


# capture background and its color properties
background = cv2.imread("backgrounds/Background_deck.jpeg")
mu_bg, sigma_inv_bg = compute_background_attributes(background)

# for loop for every card test 
for test in range(numtest):
    
    print(f"\nStarting Test{test+1}")
    
    # take starting time for benchmark evaluation
    start_time = time.perf_counter()
    
    # obtain image for the corresponding test
    img_try = cv2.imread(f"test_images/CardTest{test+1}.jpeg")
    
    # Image segmentation of foreground objects given mahalanobis distance comput2qed over dynamically acquired background
    d_M = compute_mahalanobis_distance(img_try, mu_bg, sigma_inv_bg)
    img_seg = segment_image(img_try, d_M, segment_threshold, kernel_segmentation, iterations_segmentation)
    
    # Filter out background pixels, keeping only the foreground
    img_try = cv2.bitwise_and(img_try, img_try, mask=img_seg)

    # Save filtered out blob detected in a list of Blob object
    numlabels, img_blob, stats, centroids = detect_blobs(img_seg)
    blobs = [
        Blob(x, y, w, h, img_try[y:y + h, x:x + w])
        for x, y, w, h, _ in stats[1:]
        if w >= min_width and h >= min_height and w <= max_width and h <= max_height
    ]


    card_rec_names = []
    # For each blob in the image
    for index_blob, blob in enumerate(blobs):
        # Extract keypoints and descriptors
        keypoints, descriptors = extract_keypoints_and_descriptors(blob.image, max_key_target, False, edge_threshold, detector)
        
        # Initialize accumulator array for counting robust ransac homography card matches
        index_count = np.zeros(len(deck), dtype=int)

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
        
        # Normalization of index count by the highest count to compute relative confidence
        max_count = max(index_count)
        index_count_rel1 = index_count
        if max_count > 0:
            index_count_rel1 = index_count / max_count
        
             
        # Add card names to matched_cards for specific blob
        for i in range(len(index_count)):
            # If the relative confidence and match count is above their threshold
            if index_count_rel1[i] >= confidence and index_count[i] > abs_thresh1:
                
                
                # appends the name of the card found
                if card_names[deck[i].ID] not in card_rec_names:
                    # add one of one card is found in general
                    found_cards += 1
                    card_rec_names.append(card_names[deck[i].ID])
                    
                    # if the card is really in the benchmark array 
                    if card_names[deck[i].ID] in Test_True_Cards[test]:  
                        # add one if the right card has been found
                        found_true_cards += 1
                        
                        # system to get averages and minumim from algorithm confidence    
                        print(f"{card_names[deck[i].ID]},", f"Confidence: {int(index_count_rel1[i]*100)}%", 'OK')

                        right_confidences += int(index_count_rel1[i]*100)
                        if min_right_confidence > int(index_count_rel1[i]*100):
                            min_right_confidence = int(index_count_rel1[i]*100)
                    else:
                        # system to get averages and minumim from algorithm confidence
                        print(f"{card_names[deck[i].ID]},", f"Confidence: {int(index_count_rel1[i]*100)}%")
                        not_right_confidences += int(index_count_rel1[i]*100)
                        if max_wrong_confidence < int(index_count_rel1[i]*100):
                            max_wrong_confidence = int(index_count_rel1[i]*100)


    # get time performances and saves them
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_times.append(elapsed_time)

    # true cards for the actual test
    true_cards = Test_True_Cards[test]

    # check to see whether a true card has been found 
    for i in range(len(true_cards)):
        if true_cards[i] not in card_rec_names:
            not_found_cards += 1
            print(f"{true_cards[i]} not found")
            
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

# ================================================
# 9. Benchmark printout
# ================================================

# print performances values
print(f"\nTotal cards found wrt total images {found_cards}\{total_cards}")
print(f"Right cards found wrt total images {found_true_cards}\{total_cards}")
print(f"Cards not found wrt total images {not_found_cards}\{total_cards}")
print("\nElapsed average time:", np.round(np.mean(elapsed_times), 3))
print("Average frequency:", np.round(1/np.mean(elapsed_times), 3))
print("Average right confidency", np.round(right_confidences/found_true_cards, 2))

if found_cards == found_true_cards:
    print("Average wrong confidency", 0)
else:
    print("Average wrong confidency", np.round(not_right_confidences/(found_cards-found_true_cards), 2))
print("Min right confidency", min_right_confidence)
print("Max wrong confidency", max_wrong_confidence)

# ================================================
# 10. Benchmark saving
# ================================================

# Saves parameters in the dictionary
parameters['segment_threshold'] = segment_threshold
parameters['kernel_segmentation'] = kernel_segmentation
parameters['iterations_segmentation'] = iterations_segmentation

parameters['max_key_deck'] = max_key_deck
parameters['max_key_target'] = max_key_target
parameters['edge_threshold'] = edge_threshold

parameters['lowe_ratio1'] = lowe_ratio1
parameters['lowe_ratio2'] = lowe_ratio2
parameters['matcher'] = matcher

parameters['confidence'] = confidence
parameters['abs_thresh'] = abs_thresh1

parameters['detector'] = detector

# Saves results in the dictionary
results['Number of tests'] = len(Test_True_Cards)
results['total_cards'] = total_cards
results['found_cards'] = found_cards
results['found_right_cards'] = found_true_cards
results['not_found_cards'] = not_found_cards
results['elapsed_times'] = np.round(np.mean(elapsed_times), 3)
results['frequency'] =  np.round(1/np.mean(elapsed_times), 3)
results['relative_rec_cards'] =np.round(found_cards/total_cards, 2)*100
results['relative_right_cards'] = np.round(found_true_cards/total_cards, 2)*100
results['relative_notfound_cards'] = np.round(not_found_cards/total_cards, 2)*100
results['right cards average confidence'] = np.round(right_confidences/found_true_cards, 2)

if found_cards == found_true_cards:
    results['wrong cards average confidence'] = 0
else:
    results['wrong cards average confidence'] = np.round(not_right_confidences/(found_cards-found_true_cards), 2)
    
results['Min right confidence'] = min_right_confidence
results['Max Wrong confidence'] = max_wrong_confidence

# Create DataFrames for the current run
results_df = pd.DataFrame([results], index=label)
parameters_df = pd.DataFrame([parameters], index=label)

# Define the path to the output Excel file
output_path = 'results.xlsx'

# Check if the Excel file already exists
if os.path.exists(output_path):
    # Load the existing Excel file with the 'openpyxl' engine
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        # Check if the 'Results' sheet exists
        if 'Results' in writer.sheets:
            # Append the new results at the end of the existing data
            existing_results_df = pd.read_excel(output_path, sheet_name='Results', engine='openpyxl', index_col=0)
            combined_results_df = pd.concat([existing_results_df, results_df])
            combined_results_df.to_excel(writer, sheet_name='Results', index=True)
        else:
            # If the 'Results' sheet does not exist, create it with the new data
            results_df.to_excel(writer, sheet_name='Results', index=True)
        
        # Check if the 'Parameters' sheet exists
        if 'Parameters' in writer.sheets:
            # Append the new parameters at the end of the existing data
            existing_parameters_df = pd.read_excel(output_path, sheet_name='Parameters', engine='openpyxl', index_col=0)
            combined_parameters_df = pd.concat([existing_parameters_df, parameters_df])
            combined_parameters_df.to_excel(writer, sheet_name='Parameters', index=True)
        else:
            # If the 'Parameters' sheet does not exist, create it with the new data
            parameters_df.to_excel(writer, sheet_name='Parameters', index=True)
else:
    # If the file doesn't exist, create a new one with the current run data
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Results', index=True)
        parameters_df.to_excel(writer, sheet_name='Parameters', index=True)

print(f"Data has been saved to {output_path}")
