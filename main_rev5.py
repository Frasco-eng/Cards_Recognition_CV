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

def print_image(img, colormap=cv2.COLOR_BGR2RGB, title=None):
    '''
    Print the given image with a specific colormap
    '''
    plt.title(title)
    plt.axis("off")
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


def segment_image(img, distances, threshold, kernel_size=3, iterations=5):
    '''
    Segment the given image and dilate the map
    '''
    binary_mask = np.where(distances < threshold, 0, 255).astype(np.uint8) # Create a binary mask for identify only object differently from bg
    binary_mask = binary_mask.reshape(img.shape[0:2])
    img_dil = cv2.dilate(binary_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations) # Dilate the image to make image most likely uniform
    return img_dil


def detect_blobs(img, connectivity=8): 
    '''
    Return the blobs with stats of the given segmented image
    '''
    return cv2.connectedComponentsWithStats(img, None, connectivity, cv2.CV_32S)


def show_bounding_boxes(img, blobs):
    '''
    Print the bounding box of the blobs in image
    '''
    img_blob_boxes = img.copy()
    
    for blob in blobs:
        startpoint = (blob.x, blob.y)
        endpoint = (startpoint[0] + blob.width, startpoint[1] + blob.height) 
        cv2.rectangle(img_blob_boxes, startpoint, endpoint, color=(255, 0, 0), thickness=4) 

    print_image(img_blob_boxes)
    
    return img_blob_boxes


def extract_keypoints_and_descriptors(img, max_key, draw_key_points=False): 
    '''
    Return the most max_key significant keypoints and descriptors of a given image  
    '''
    sift = cv2.SIFT_create(edgeThreshold=5)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True) # Only consider the most significant descpritor
    max_keypoints = max_key
    keypoints = keypoints[: max_keypoints] 
    descriptors = descriptors[: max_keypoints] if descriptors is not None else None

    if draw_key_points:
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print_image(img_with_keypoints)

    return keypoints, descriptors


# Creazione del mazzo
def create_deck(input_path, output_path, num_cards, k_size, iter):
    '''
    Create the deck in the specified output_path
    '''
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
        threshold = 60

        count = 1
        for i in range(1, num_cards + 1):
            img = cv2.imread(f"{input_path}{i}.png")
            distances = compute_mahalanobis_distance(img, mu, sigma_inv)
            segmented = segment_image(img, distances, threshold, k_size, iter)
            _, _, stats, _ = detect_blobs(segmented)

            for label in range(1, stats.shape[0]):
                x, y, w, h, _ = stats[label]
                cropped_img = img[y:y + h, x:x + w]
                cv2.imwrite(f"{output_path}{count}.png", cropped_img)
                count += 1
        
        print("Deck created!")
        return False


def compute_deck_features(deck_path, max_key):
    '''
    Return the keypoints and descriptor for each card in the deck directory
    '''
    deck = []
    for img_file in os.listdir(deck_path):
        img = cv2.imread(os.path.join(deck_path, img_file))
        keypoints, descriptors = extract_keypoints_and_descriptors(img, max_key)
        deck.append(Card(img, keypoints, descriptors))
    return deck



########################################
segment_threshold = 60
kernel_segmentation = 3
iterations_segmentation = 5

# Deck creation
images_path = "images"
deck_path = "deck"
cards_number = 38

#create_deck(images_path + "/Card", deck_path + "/Card", cards_number, kernel_segmentation, iterations_segmentation)

img_bg = cv2.imread("images/Background.png")
img_try = cv2.imread("images/Card1over.jpeg")

mu_bg, sigma_inv_bg = compute_background_attributes(img_bg)


#print("background attributes : " + str(mu_bg) + ", " + str(sigma_inv_bg))

d_M = compute_mahalanobis_distance(img_try, mu_bg, sigma_inv_bg)

#print("mahalanobis distance: " + str(d_M))

img_seg = segment_image(img_try, d_M, segment_threshold, kernel_segmentation, iterations_segmentation)

#print_image(img_seg)

numlabels, img_blob, stats, centroids = detect_blobs(img_seg)

#print(numlabels, img_blob, stats, centroids)
# Lower bound for pixel card blob for the current camera used!!!!!!!!!!!!!!!!!!!!!!
min_width = 50
min_height = 80

blobs = [
    Blob(x, y, w, h, img_try[y:y + h, x:x + w]) 
    for x, y, w, h, _ in stats[1:] 
    if w >= min_width and h >= min_height
]

show_bounding_boxes(img_try, blobs)



# Compute deck features (images, keypoints, descriptors) for each cards in deck
max_key = 5000
deck = compute_deck_features(deck_path, max_key)

# Crea l'oggetto BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

index_count = np.zeros(len(deck), dtype=int)

bin_size = 4
lowe_ratio1=0.8
lowe_ratio2=0.8
vote_threshold = 10  # Example threshold

vote_space = np.zeros((bin_size, bin_size, bin_size, bin_size))
#vote_space = np.zeros((bin_size, bin_size))

# Define ranges for parameters (example ranges)
scale_range = (0.1, 10)
angle_range = (0, 360)
tx_range = (-2000, 2000)
ty_range = (-2000, 2000)

# Compute bin width for each dimension
scale_bin_width = (scale_range[1] - scale_range[0]) / bin_size
angle_bin_width = (angle_range[1] - angle_range[0]) / bin_size
tx_bin_width = (tx_range[1] - tx_range[0]) / bin_size
ty_bin_width = (ty_range[1] - ty_range[0]) / bin_size

for blob in blobs:
    query_matches = []
    keypoints, descriptors = extract_keypoints_and_descriptors(blob.image, max_key, False)
    for i in range(len(deck)):
        matches = bf.knnMatch(descriptors, deck[i].descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < lowe_ratio1 * n.distance:
                # Verifica se la corrispondenza è reciproca
                reverse_matches = bf.knnMatch(deck[i].descriptors[m.trainIdx].reshape(1, -1), descriptors, k=2)
                for p, q in reverse_matches:
                    if p.distance < lowe_ratio2 * q.distance:
                        if reverse_matches[0][0].trainIdx == m.queryIdx:
                            good_matches.append(m)
                            #index_count[i]+=1
                            query_matches.append(m.queryIdx)

        matched_keypoints = []

        count=0
        for match in good_matches:
            
            # Iterate through matches
            kp_template = deck[i].keypoints[match.trainIdx]
            kp_target = keypoints[match.queryIdx]

            # Parameters
            #print(kp_template.angle, kp_target.angle)
                
            angle = -kp_template.angle + kp_target.angle # Relative rotation
            if angle<0:
                angle = 360 + angle

                
            scale = kp_template.size/ kp_target.size   # Relative scale

            # Compute translation with rotation: is this what we need to do?
            tx = kp_target.pt[0] - scale * (kp_template.pt[0] * np.cos(np.radians(angle)) - kp_template.pt[1] * np.sin(np.radians(angle)))
            ty = kp_target.pt[1] - scale * (kp_template.pt[0] * np.sin(np.radians(angle)) + kp_template.pt[1] * np.cos(np.radians(angle)))

            
            M = np.array([
            [scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)), tx],
            [scale * np.sin(np.radians(angle)), scale * np.cos(np.radians(angle)), ty]
            ], dtype=np.float32)
            
            # Apply the transformation to the template image
            height, width = deck[i].image.shape[:2]  # Get the dimensions of the template image
            transformed_template = cv2.warpAffine(deck[i].image, M, (width, height), flags=cv2.INTER_LINEAR)


            # Map parameters to bins
            scale_bin = int((scale - scale_range[0]) / scale_bin_width)
            angle_bin = int((angle - angle_range[0]) / angle_bin_width)
            tx_bin = int((tx - tx_range[0]) / tx_bin_width)
            ty_bin = int((ty - ty_range[0]) / ty_bin_width)
        
            # Ensure bins are within bounds
            if 0 <= scale_bin < bin_size and 0 <= angle_bin < bin_size and 0 <= tx_bin < bin_size and 0 <= ty_bin < bin_size:
                vote_space[scale_bin, angle_bin, tx_bin, ty_bin] += 1
                #vote_space[scale_bin, angle_bin] += 1
                matched_keypoints.append((match, (scale_bin, angle_bin, tx_bin, ty_bin)))
                #matched_keypoints.append((match, (scale_bin, angle_bin)))
                    
        filtered_keypoints = []
        
        significant_bins = np.argwhere(vote_space >= vote_threshold)
        
        # Retain only keypoints contributing to significant bins
        for match, bins in matched_keypoints:
            if tuple(bins) in [tuple(bin_indices) for bin_indices in significant_bins]:
                
                
                filtered_keypoints.append(match)
                
        #query_keypoints = [keypoints[match.queryIdx] for match in filtered_keypoints]
        #print_image(cv2.drawKeypoints(blob.image, query_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS), title="Good Matches")
                
        index_count[i] += len(filtered_keypoints)
    


print(f"Index count: {index_count}")

# !!!!! dynamic computation of thresh_choice relative to max_key !!!!!
print_num = 5
sorted_index = np.argsort(index_count)[::-1]
sorted_index = sorted_index[0:print_num]
for i in range(len(sorted_index)):
    if sorted_index[i] > 0: 

        print_image(deck[sorted_index[i]].image, title=f"Guessed card Counted: {index_count[sorted_index[i]]}")
        
        
        #test_indexes = [m.trainIdx - max_key*i for m in good_matches if m.trainIdx >= max_key*i and m.trainIdx < max_key*(i+1)]
        #print(len(deck[i].keypoints))
        #test_keypoints = [deck[i].keypoints[j] for j in test_indexes]
        #deck_image_keypoints = cv2.drawKeypoints(deck[i].image, test_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)