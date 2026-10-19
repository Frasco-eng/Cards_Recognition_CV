import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import re
import pandas as pd
import openpyxl
from multiprocessing import Pool

card_names = {1:"1_CUPS", 2:"2_CUPS", 3:"3_CUPS", 4:"4_CUPS", 5:"5_CUPS", 6:"6_CUPS", 7:"7_CUPS", 8:"KNAVE_CUPS", 9:"KNIGHT_CUPS", 10:"KING_CUPS", 
              11:"1_COINS", 12:"2_COINS", 13:"3_COINS", 14:"4_COINS", 15:"5_COINS", 16:"6_COINS", 17:"7_COINS", 18:"KNAVE_COINS", 19:"KNIGHT_COINS", 20:"KING_COINS",
              21:"1_SWORDS", 22:"2_SWORDS", 23:"3_SWORDS", 24:"4_SWORDS", 25:"5_SWORDS", 26:"6_SWORDS", 27:"7_SWORDS", 28:"KNAVE_SWORDS", 29:"KNIGHT_SWORDS", 30:"KING_SWORDS", 
              31:"1_BATONS", 32:"2_BATONS", 33:"3_BATONS", 34:"4_BATONS", 35:"5_BATONS", 36:"6_BATONS", 37:"7_BATONS", 38:"KNAVE_BATONS", 39:"KNIGHT_BATONS", 40:"KING_BATONS"}  
              
              
class Blob:
    def __init__(self, x, y, width, height, image):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.image = image

class Card:
    def __init__(self, image, ID, keypoints, descriptors):
        self.image = image
        self.ID = ID
        self.keypoints = keypoints
        self.descriptors = descriptors
        

    def draw_keypoints(self):
        return cv2.drawKeypoints(self.image, self.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



def print_image(img, colormap=cv2.COLOR_BGR2RGB, title=None):
    plt.title(title)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, colormap))
    plt.show()

def compute_background_attributes(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    mu = np.mean(img_rgb, axis=0)
    sigma_inv = np.linalg.inv(np.cov(img_rgb.T))
    return mu, sigma_inv

def compute_mahalanobis_distance(img, mu, sigma_inv):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    diff = img_rgb - mu
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, sigma_inv, diff))

def segment_image(img, distances, threshold, kernel_size=3, iterations=5):
    binary_mask = (distances >= threshold).astype(np.uint8) * 255
    binary_mask = binary_mask.reshape(img.shape[:2])
    return cv2.dilate(binary_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)

def detect_blobs(img, connectivity=8):
    return cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)

def show_bounding_boxes(img, blobs):
    img_blob_boxes = img.copy()
    for blob in blobs:
        startpoint = (blob.x, blob.y)
        endpoint = (startpoint[0] + blob.width, startpoint[1] + blob.height)
        cv2.rectangle(img_blob_boxes, startpoint, endpoint, color=(255, 0, 0), thickness=4)
    print_image(img_blob_boxes)
    return img_blob_boxes

def extract_keypoints_and_descriptors(img, max_key, draw_key_points, edge_threshold, detector):
    if detector == 'sift':
        sift = cv2.SIFT_create(edgeThreshold=edge_threshold)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_key]
        descriptors = descriptors[:max_key] if descriptors is not None else None
    if detector == 'orb':
        orb = cv2.ORB_create(nfeatures = max_key)
        keypoints, descriptors = orb.detectAndCompute(img, None)
    if detector == 'akaze':
        akaze = cv2.AKAZE_create()
        keypoints, descriptors = akaze.detectAndCompute(img, None)

    if draw_key_points:
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        print_image(img_with_keypoints)

    return keypoints, descriptors

def create_deck(input_path, output_path, num_cards, k_size, iter):
    image_files = os.listdir("deck")
    if len(image_files) == cards_number:
        print("Deck already created.")
        return True

    print(f"Error: expected {cards_number} images, found {len(image_files)}.")
    print("Start deck creation...")

    img_bg = cv2.imread("test_images/Background.png")
    mu, sigma_inv = compute_background_attributes(img_bg)
    threshold = 50

    for count, i in enumerate(range(1, num_cards + 1), 1):
        img = cv2.imread(f"{input_path}{i}.png")

        if img is None:
            continue
        distances = compute_mahalanobis_distance(img, mu, sigma_inv)
        segmented = segment_image(img, distances, threshold, k_size, iter)
        _, _, stats, _ = detect_blobs(segmented)

        for x, y, w, h, _ in stats[1:]:
            cropped_img = img[y:y + h, x:x + w]
            cv2.imwrite(f"{output_path}{count}.png", cropped_img)
            count += 1

    print("Deck created!")
    return False

def compute_deck_features(deck_path, max_key, edge_threshold, detector):
    deck = []
    for img_file in os.listdir(deck_path):
        ID = re.search(r'\d+', img_file)
        img = cv2.imread(os.path.join(deck_path, img_file))
        keypoints, descriptors = extract_keypoints_and_descriptors(img, max_key, False, edge_threshold, detector)
        deck.append(Card(img, int(ID.group()), keypoints, descriptors))
    return deck






parameters = {}
results = {}

### Parameters ###

label = ['Run1']  # This will be the label for the current run

# segmentation
segment_threshold = 60
kernel_segmentation = 3
iterations_segmentation = 5

# keypoints
max_key_deck = 700
max_key_target= 5000
edge_threshold = 100
detector = 'orb'

# matching
lowe_ratio1 = 0.7
lowe_ratio2 = 0.7
matcher = 'bf'

if detector == 'orb' or detector == 'akaze':
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
if detector == 'sift':
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

index_params = dict(algorithm=1, trees=5)  # For KDTree
search_params = dict(checks=50)  # Number of checks
flann = cv2.FlannBasedMatcher(index_params, search_params)

# choosing
confidence = 0.15
abs_thresh = 5

# cards
images_path = "images"
deck_path = "deck"
cards_number = 40

# blobs
min_width = 50
min_height = 80



create_deck(images_path + "/Card", deck_path + "/Card", cards_number, kernel_segmentation, iterations_segmentation)

deck = compute_deck_features(deck_path, max_key_deck, edge_threshold, detector)
print("Deck features created")


img_bg = cv2.imread("test_images/Background.png")


numtest = 9
elapsed_times = []
found_cards = 0
found_true_cards = 0
not_found_cards=0
right_confidences = 0
not_right_confidences = 0
min_right_confidence = 100
max_wrong_confidence = 0

Test_True_Cards =   [
                    ['7_SWORDS'],
                    ['2_COINS', '1_CUPS'],
                    ['1_CUPS'],
                    ['2_BATONS', 'KNAVE_BATONS'],
                    ['7_SWORDS', 'KNIGHT_SWORDS'],
                    ['7_SWORDS', '7_BATONS'],
                    ['7_BATONS', '7_SWORDS'],
                    [],
                    ['7_BATONS', '5_SWORDS', '2_SWORDS']]
'''
['KNIGHT_COINS', '7_COINS', 'KING_CUPS'],
['4_COINS', '7_CUPS', 'KNIGHT_CUPS'],
['6_CUPS', 'KNAVE_COINS', '3_SWORDS'],
['KNAVE_BATONS', '5_BATONS', '5_SWORDS'],
['1_COINS', '2_CUPS'],
['1_SWORDS', '5_COINS', '6_BATONS'],
['2_SWORDS', 'KNIGHT_SWORDS', 'KNIGHT_BATONS'],
['3_COINS', 'KING_BATONS'],
['7_BATONS', '7_SWORDS', '2_BATONS'],
['5_BATONS', 'KNAVE_BATONS', '5_SWORDS']
'''    

for cards in Test_True_Cards:
    for card in cards:
        if card not in card_names.values():
            print(f"{card} not found in dictionary")
            exit()

total_cards = sum(len(cards) for cards in Test_True_Cards)

for test in range(numtest):
    
    print(f"\nStarting Test{test+1}")
    
    start_time = time.perf_counter()
 
    img_try = cv2.imread(f"test_images/CardTest{test+1}.jpeg")
    

    mu_bg, sigma_inv_bg = compute_background_attributes(img_bg)
    d_M = compute_mahalanobis_distance(img_try, mu_bg, sigma_inv_bg)

    img_seg = segment_image(img_try, d_M, segment_threshold, kernel_segmentation, iterations_segmentation)

    _, binary_mask = cv2.threshold(img_seg, 127, 255, cv2.THRESH_BINARY)
    img_try = cv2.bitwise_and(img_try, img_try, mask=binary_mask)
    numlabels, img_blob, stats, centroids = detect_blobs(img_seg)

    blobs = [
        Blob(x, y, w, h, img_try[y:y + h, x:x + w])
        for x, y, w, h, _ in stats[1:]
        if w >= min_width and h >= min_height
    ]

    #show_bounding_boxes(img_try, blobs)

    index_count = np.zeros(len(deck), dtype=int)

    for blob in blobs:
        auto_key_max = int(4800/250000*blob.width*blob.height-100)
        if auto_key_max > max_key_target:
            auto_key_max = max_key_target
        keypoints, descriptors = extract_keypoints_and_descriptors(blob.image, auto_key_max, False, edge_threshold, detector)
        for i, card in enumerate(deck):
            if matcher == 'bf':
                matches = bf.knnMatch(descriptors, card.descriptors, k=2)
            elif matcher == 'flann':
                matches = flann.knnMatch(descriptors, card.descriptors, k=2)

            good_matches = [(m, n) for m, n in matches if m.distance < lowe_ratio1 * n.distance]
         
            for m, n in good_matches:
                if matcher == 'bf':
                    reverse_matches = bf.knnMatch(card.descriptors[m.trainIdx].reshape(1, -1), descriptors, k=2)
                elif matcher == 'flann':
                    reverse_matches = flann.knnMatch(card.descriptors[m.trainIdx].reshape(1, -1), descriptors, k=2)

                if reverse_matches and len(reverse_matches[0]) == 2:
                    p, q = reverse_matches[0]  # Only consider the closest match
                    if p.distance < lowe_ratio2 * q.distance and reverse_matches[0][0].trainIdx == m.queryIdx:
                        index_count[i] += 1
            
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_times.append(elapsed_time)

    ### print ##
    #print(f"Index count: {index_count}")

    max_count = max(index_count)
    if max_count > 0:
        index_count_rel = index_count/max_count
    else:
        index_count_rel = index_count

    sorted_index = np.argsort(index_count)[::-1]

    card_rec_names = []

  
    for i in range(len(index_count)):
        if index_count_rel[i] >= confidence and index_count[i] > abs_thresh:
            found_cards += 1
            card_rec_names.append(card_names[deck[i].ID])
            if card_names[deck[i].ID] in Test_True_Cards[test]:  
                found_true_cards += 1
                print(f"{card_names[deck[i].ID]},", f"Confidence: {int(index_count_rel[i]*100)}%", 'OK')
                right_confidences += int(index_count_rel[i]*100)
                if min_right_confidence > int(index_count_rel[i]*100):
                    min_right_confidence = int(index_count_rel[i]*100)
            else:
                print(f"{card_names[deck[i].ID]},", f"Confidence: {int(index_count_rel[i]*100)}%")
                not_right_confidences += int(index_count_rel[i]*100)
                if max_wrong_confidence < int(index_count_rel[i]*100):
                    max_wrong_confidence = int(index_count_rel[i]*100)

    true_cards = Test_True_Cards[test]

    for i in range(len(true_cards)):
        if true_cards[i] not in card_rec_names:
            not_found_cards += 1
            print(f"{true_cards[i]} not found")
            
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

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
parameters['abs_thresh'] = abs_thresh

parameters['detector'] = detector

 
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
