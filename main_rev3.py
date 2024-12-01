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
    def __init__(self, image):
        self.image = image
        self.R_table = None

    def set_R_table(self, R_table):
        self.R_table = R_table



def show_image(img, colormap=cv2.COLOR_BGR2RGB, title=None):
    '''
    Print the given image with a specific colormap
    '''
    plt.title(title)
    plt.axis("off")
    if colormap == 'gray':
        plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
    else:
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

    show_image(img_blob_boxes)
    
    return img_blob_boxes


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
        threshold = 2

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


def compute_deck(deck_path):
    deck = []
    for img_file in os.listdir(deck_path):
        img = cv2.imread(os.path.join(deck_path, img_file), cv2.IMREAD_GRAYSCALE)
        deck.append(Card(img))
    return deck


########################################
segment_threshold = 50
kernel_segmentation = 3
iterations_segmentation = 5

# Deck creation
images_path = "images"
deck_path = "deck"
cards_number = 38

#create_deck(images_path + "/Card", deck_path + "/Card", cards_number, kernel_segmentation, iterations_segmentation)

img_bg = cv2.imread("images/Background.png")
target_img = cv2.imread("images/Card9.png")

mu_bg, sigma_inv_bg = compute_background_attributes(img_bg)

d_M = compute_mahalanobis_distance(target_img, mu_bg, sigma_inv_bg)

img_seg = segment_image(target_img, d_M, segment_threshold, kernel_segmentation, iterations_segmentation)
# print_image(img_seg)

numlabels, img_blob, stats, centroids = detect_blobs(img_seg)

# Lower bound for pixel card blob for the current camera used!!!!!!!!!!!!!!!!!!!!!!
min_width = 50
min_height = 80

blobs = [
    Blob(x, y, w, h, target_img[y:y + h, x:x + w]) 
    for x, y, w, h, _ in stats[1:] 
    if w >= min_width and h >= min_height
]

#show_bounding_boxes(target_img, blobs)


deck = compute_deck(deck_path)


Canny_thresholdL = 100
Canny_thresholdH = 200
quantization_step = np.pi / 12  # Example: 30-degree bins
nsample = 100

### OFFLINE ###

for card in deck:
    
    # 1. Detect edges in the model image (grayscaled)
    edges = cv2.Canny(card.image, Canny_thresholdL, Canny_thresholdH)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overall_m10, overall_m01, overall_m00 = 0, 0, 0

    for contour in contours:
        moments = cv2.moments(contour)
        overall_m10 += moments["m10"]
        overall_m01 += moments["m01"]
        overall_m00 += moments["m00"]
        
    if overall_m00 != 0:
        Bx = int(overall_m10 / overall_m00)
        By = int(overall_m01 / overall_m00)
        reference_point = (Bx, By)
        
    #plt.imshow(edges, cmap='gray', vmin = 0, vmax = 255)
    #plt.scatter(Bx, By, color='red', s=50, label='B') 
    #plt.show()

    # 3.1 Compute vectors r from edges x on corners to B 
    edge_points = np.column_stack(np.where(edges > 0))
    
    sample_indices = np.random.choice(len(edge_points), nsample, replace=False)
    selected_edge_points = edge_points[sample_indices]

    # Compute r_vectors for the selected subset
    r_vectors = reference_point - selected_edge_points 

    # 3.2 Compute gradient direction phi(x)
    grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=7)
    grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=7)
    phi = np.arctan2(grad_y, grad_x)  # Gradient direction in radians

    '''
    step = 3  # Set step size for the grid

    for y in range(0, edges.shape[0], step):
        for x in range(0, edges.shape[1], step):
            # Calculate dx and dy based on gradient direction phi
            if edges[y, x] > 0:
                dx = int(10 * np.cos(phi[y, x]))
                dy = int(10 * np.sin(phi[y, x]))
                
                # Draw the arrow on the image (use (x, y) as start and (x + dx, y + dy) as end)
                cv2.arrowedLine(card.image, (x, y), (x + dx, y + dy), (0, 255, 0),1, tipLength=0.3)

    # Display the image
    show_image(card.image)
    plt.show()
    '''
    # Compute gradient direction phi(x) for the selected subset

    phi_subset = phi[selected_edge_points[:, 0], selected_edge_points[:, 1]]
    # 3.3 Store r in a table (R-Table) indexed by a chosen gradient's direction quantization step

    phi_quantized = np.floor(phi_subset / quantization_step).astype(int) % (int(2*np.pi / quantization_step))
    
    R_table = {}
    for r, quant in zip(r_vectors, phi_quantized):
        if quant not in R_table:
            R_table[quant] = []
        R_table[quant].append(r)

    card.set_R_table(R_table)

print("Deck Complete")

nsample = 100


### ONLINE ###
score = np.zeros(len(deck), dtype=np.int32)
for blob in blobs:
    # 1. Detect edges in the target image and initialize another image as accumulator array A[y]
    edges = cv2.Canny(blob.image, Canny_thresholdL, Canny_thresholdH)

    
    edge_points = np.column_stack(np.where(edges > 0))
    sample_indices = np.random.choice(len(edge_points), nsample, replace=False)
    selected_edge_points = edge_points[sample_indices]
    
    # 2. Compute gradient direction phi(x)
    grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=7)
    grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=7)
    phi_target = np.arctan2(grad_y, grad_x)

    step = 3  # Set step size for the grid

    for y in range(0, edges.shape[0], step):
        for x in range(0, edges.shape[1], step):
            # Calculate dx and dy based on gradient direction phi
            if edges[y, x] > 0:
                dx = int(10 * np.cos(phi[y, x]))
                dy = int(10 * np.sin(phi[y, x]))
                
                # Draw the arrow on the image (use (x, y) as start and (x + dx, y + dy) as end)
                cv2.arrowedLine(blob.image, (x, y), (x + dx, y + dy), (0, 255, 0),1, tipLength=0.3)

    # Display the image
    show_image(blob.image)
    plt.show()


    phi_subset = phi_target[selected_edge_points[:, 0], selected_edge_points[:, 1]]
    
    phi_target_quantized = np.floor(phi_subset / quantization_step).astype(int) % (int(2*np.pi / quantization_step))

    # 3. Quantize phi to index the R-Table. For each r_i vector stored into the indexed row:
    
    num_theta_steps = 36  # Number of steps, e.g., 10 degrees per step
    accumulator_steps = 20
    for i in range(len(deck)):
        
        accumulator = np.zeros((target_img.shape[0] // accumulator_steps, target_img.shape[1] // accumulator_steps), dtype=np.int32)
        for theta_idx in range(num_theta_steps):
            theta = (theta_idx * (2 * np.pi / num_theta_steps))
            for edge_point, quant in zip(selected_edge_points, phi_target_quantized):
                if quant in deck[i].R_table:
                    for r in deck[i].R_table[quant]:
                        # 3.1 theta and s shall be discretize (how?)
                        s = 1     # Example: Scale invariance

                        # 3.2 Compute the position the reference point y: y = x+ s*r_theta
                        y, x = edge_point
                        ref_point_y = int(y + s * (r[0] * np.cos(theta) + r[1] * np.sin(theta)))
                        ref_point_x = int(x + s * (-r[0] * np.sin(theta) + r[1] * np.cos(theta)))
                        
                        bin_x = ref_point_x // accumulator_steps
                        bin_y = ref_point_y // accumulator_steps
                        
                        # 3.3 Cast a vote into the accumulator array A[y]++
                        if 0 <= ref_point_x < accumulator.shape[0] and 0 <= ref_point_y < accumulator.shape[1]:
                            accumulator[bin_y, bin_x] += 1
                        
        # 4. Find the score for the current card
        
        score[i] = np.sum(accumulator)  # Count the number of significant votes
        
print(score)
print_num = 3
sorted_index = np.argsort(score)[::-1]
sorted_index = sorted_index[0:print_num]
for i in range(len(sorted_index)):
    if score[sorted_index[i]] > 0: 
        show_image(deck[sorted_index[i]].image, title=f"Guessed card Counted: {score[sorted_index[i]]}")
        


