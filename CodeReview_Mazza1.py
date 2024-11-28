### Imported libraries ###

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


# Blobs: rappresenta una regione dell'immagine segmentata
class Blob:
    def __init__(self, x, y, width, height, image):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.image = image


# Cards: contiene l'immagine della carta e i relativi descrittori
class Card:
    def __init__(self, image, keypoints, descriptors):
        self.image = image
        self.keypoints = keypoints
        self.descriptors = descriptors

    def draw_keypoints(self):
        return cv2.drawKeypoints(self.image, self.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


'''
AXIS USAGE
    axis=0: L'operazione viene eseguita lungo le colonne (o il primo asse).
    axis=1: L'operazione viene eseguita lungo le righe (o il secondo asse).
    axis=2: L'operazione viene eseguita lungo il terzo asse, tipico di immagini RGB o array multidimensionali.
    axis=-1: Si riferisce all'ultimo asse, indipendentemente dalla dimensionalità.

'''

def print_image(img, colormap=None):
    plt.imshow(cv2.cvtColor(img, colormap))
    plt.show()

def compute_background_attributes(img):
    '''
    Compute the mean and inverse of covariance matrix of the given image
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3) # Change of shape: (45, 42, 3) -> (1890, 3). And color space: BGR -> RGB
    mu = np.mean(img_rgb, axis=0) # Computing mean across channels
    sigma_inv = np.linalg.inv(np.cov(img_rgb.T)) # Computing inverse of covariance matrix
    return mu, sigma_inv


def compute_mahalanobis_distance(img, mu, sigma_inv):
    '''
    Compute the mahalanobis distance for each pixel in the image
    '''
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    diff = img_rgb - mu
    return np.sqrt(np.sum((diff @ sigma_inv) * diff, axis=1)) # Compute sum of each pixel


def segment_image(img, distances, threshold):
    binary_mask = np.where(distances < threshold, 0, 255).astype(np.uint8) # Create a binary mask for identify only object differently from bg
    return binary_mask.reshape(img.shape[0:2])


def detect_blobs(img, kernel_size, iterations): 
    img_dil = cv2.dilate(img, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations) # Dilate the image to make image most likely uniform
    return cv2.connectedComponentsWithStats(img_dil, None, 8, cv2.CV_32S)


def show_bounding_boxes(img, blobs):
    img_blob_boxes = img.copy()
    
    for blob in blobs:
        startpoint = (blob.x, blob.y)
        endpoint = (startpoint[0] + blob.width, startpoint[1] + blob.height) 
        cv2.rectangle(img_blob_boxes, startpoint, endpoint, color=(255, 0, 0), thickness=4) 

    print_image(img_blob_boxes, cv2.COLOR_BGR2RGB)
    
    return img_blob_boxes


def extract_keypoints_and_descriptors(img, max_key, draw_key_points=False): 
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True) # Only consider the most significant descpritor
    max_keypoints = max_key
    keypoints = keypoints[: max_keypoints] 
    descriptors = descriptors[: max_keypoints] if descriptors is not None else None

    if draw_key_points:
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print_image(img_with_keypoints, cv2.COLOR_BGR2RGB)

    return keypoints, descriptors


# Creazione del mazzo
def create_deck(input_path, output_path, num_cards):

    image_files = [f for f in os.listdir("deck")]
    actual_cards_number = len(image_files)

    if actual_cards_number == cards_number:
        print("Deck already created.")
        return True
    else:
        print(f"Error: expected {cards_number} images, found {actual_cards_number}.")
        print("Start deck creation...")

        img_bg = cv2.imread("images/Background.png")
        mu, sigma_inv = compute_background_attributes(img_bg)
        threshold = 50
        count = 1

        for i in range(1, num_cards + 1):
            img = cv2.imread(f"{input_path}{i}.png")
            distances = compute_mahalanobis_distance(img, mu, sigma_inv)
            segmented = segment_image(img, distances, threshold)
            _, _, stats, _ = detect_blobs(segmented, 3, 5)

            for label in range(1, stats.shape[0]):
                x, y, w, h, _ = stats[label]
                cropped_img = img[y:y + h, x:x + w]
                cv2.imwrite(f"{output_path}{count}.png", cropped_img)
                count += 1
        
        return False


def compute_deck_features(deck_path, max_key):
    deck = []
    for img_file in os.listdir(deck_path):
        img = cv2.imread(os.path.join(deck_path, img_file))
        keypoints, descriptors = extract_keypoints_and_descriptors(img, max_key)
        deck.append(Card(img, keypoints, descriptors))
    return deck


########################################
img_bg = cv2.imread("images/Background.png")
img_try = cv2.imread("images/Card2_over2.jpg")

bg_attr = compute_background_attributes(img_bg)

d_M = compute_mahalanobis_distance(img_try, bg_attr[0], bg_attr[1])

segment_threshold = 50

img_seg = segment_image(img_try, d_M, segment_threshold)

print_image(img_seg, cv2.COLOR_BGR2RGB)

numlabels, img_blob, stats, centroids = detect_blobs(img_seg, 3, 5)

blobs = [Blob(x, y, w, h, img_try[y:y + h, x:x + w]) for x, y, w, h, _ in stats[1:]]

show_bounding_boxes(img_try, blobs)

# Deck creation
images_path = "images"
deck_path = "deck"
cards_number = 38

create_deck(images_path + "/Card", deck_path + "/Card", cards_number)

# Compute deck features (images, keypoints, descriptors) for each cards in deck
max_key = 500
deck = compute_deck_features(deck_path, max_key)


### Cards recognition process ###                           --> TO MODIFY

# FLANN Paramters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks = 50)

# FLANN Creation
flann = cv2.FlannBasedMatcher(index_params, search_params)

all_descriptors = np.vstack([card.descriptors for card in deck if card.descriptors is not None])
index_matches = []

for blob in blobs:
    keypoints, descriptors = extract_keypoints_and_descriptors(blob.image, max_key, False)
    matches = flann.knnMatch(descriptors, all_descriptors.reshape(-1,128).astype(np.float32), k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            index_matches.append(int(np.floor(m.trainIdx/max_key)))

print(index_matches)

index_count = np.zeros((len(deck), 1))

for i in range(len(index_matches)):
    index_count[index_matches[i]] += 1

thresh_choice = 5
index_count[index_count < thresh_choice] = 0

for i in range(len(index_count)):
    if index_count[i] > 0:
        print_image(deck[i].image, cv2.COLOR_BGR2RGB)