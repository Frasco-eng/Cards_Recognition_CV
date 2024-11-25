import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

## CLASSES ##
## CLASSES ##
## CLASSES ##

class Blobs(object):
    def __init__(self, x_0, y_0, width, height, image):
        self.x_0 = x_0
        self.y_0 = y_0
        self.width = width
        self.height = height
        self.image = image       

class Cards(object):
    def __init__(self, image, keypoints, descriptors):
        self.image = image  
        self.keypoints = keypoints
        self.descriptors = descriptors

    def get_card_with_keys(self):
        return cv2.drawKeypoints(self.image.copy(), self.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


## FUNCTIONS ##
## FUNCTIONS ##
## FUNCTIONS ##

def background_attributes(img):
    mu = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3), axis=0) # mean for all the pixels on each channel
    sigma = np.cov(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3).T)     # ha i 3 canali sulle row, sulle colonne le intensità dei pixel per canale 
    sigma_inv = np.linalg.inv(sigma)
    return mu, sigma_inv

def mahalanobis(img, attr):
    mu = attr[0]
    sigma_inv = attr[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    mu = mu * np.ones(img.reshape(-1, 3).shape)
    return np.sqrt(np.sum(np.multiply((img - mu) @ sigma_inv, (img - mu)), axis=1))

def segment(img, d_M, thresh):
    img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(-1)
    img_copy[d_M < thresh] = 0
    img_copy[d_M >= thresh] = 255
    return img_copy.reshape(img.shape[0:2])
    
def get_blobs(img, kernel_size, iterations):
    img_dil = cv2.dilate(img, np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations)
    return cv2.connectedComponentsWithStats(img_dil, None, 8, cv2.CV_32S)

def key_and_desc(img):
    sift = cv2.SIFT_create()                    #nfeatures=200
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    max_keypoints = 100
    keypoints = keypoints[: max_keypoints] 
    descriptors = descriptors[: max_keypoints] if descriptors is not None else None
    return keypoints, descriptors

def create_deck(path1, path2, num_cards):
    img_bg = cv2.imread("images/Background.png")
    bg_attr = background_attributes(img_bg)
    thresh=50
    count=1

    for i in range(1,num_cards+1):
        path = path1 + str(i) + ".png"
        img = cv2.imread(path)
        d_M = mahalanobis(img, bg_attr)
        img_seg = segment(img, d_M, thresh)
        numlabels, img_blob, stats, centroids = get_blobs(img_seg, 3, 5)
        cards = []
        for labels in range(1, numlabels):
            startpoint = (stats[labels, 0],stats[labels, 1])
            endpoint = (startpoint[0]+ stats[labels, 2], startpoint[1]+ stats[labels, 3]) 
            img_cropped = img[startpoint[1]:endpoint[1], startpoint[0]:endpoint[0], :]
            blob = Blobs(startpoint[1], startpoint[0], stats[labels, 2], stats[labels, 3], img_cropped)
            #cards.append(blob)
            cv2.imwrite(path2 +str(count)+".png", blob.image)
            count += 1

def deck_compute_features_and_descriptors(path):
    deck=[]
    for image in os.listdir(path):
        img_path = os.path.join(path, image)
        img = cv2.imread(img_path)
        keypoints, descriptors = key_and_desc(img)
        card = Cards(img, keypoints, descriptors)
        deck.append(card)
    return deck
        
    
## PROCESS ##
## PROCESS ##
## PROCESS ##

'''
img_bg = cv2.imread("images/Background.png")
img1 = cv2.imread("images/Card22.png")

d_M = mahalanobis(img1, background_attributes(img_bg))
thresh = 50

img1_seg = segment(img1, d_M, thresh)

numlabels, img1_blob, stats, centroids = get_blobs(img1_seg, 3, 5)  # stats = [left, top, width, height, area]

cards = []

for labels in range(1, numlabels):
    startpoint = (stats[labels, 0],stats[labels, 1])
    endpoint = (startpoint[0]+ stats[labels, 2], startpoint[1]+ stats[labels, 3]) 
    img_cropped = img1[startpoint[1]:endpoint[1], startpoint[0]:endpoint[0], :]
    blob = Blobs(startpoint[1], startpoint[0], stats[labels, 2], stats[labels, 3], img_cropped)
    cards.append(blob)

#cv2.rectangle(img1, startpoint, endpoint, thickness=2, color=(255, 0, 0))
#plt.imshow(img1[startpoint[1]:endpoint[1], startpoint[0]:endpoint[0], :]

keypoints, descriptors = key_and_desc(cards[0].image)

print(descriptors.shape)
img_with_keypoints = cv2.drawKeypoints(cards[0].image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
plt.show()
'''

#create_deck("images/Card", "deck/Card", 38)
deck=deck_compute_features_and_descriptors("deck")
#plt.imshow(cv2.cvtColor(deck[1].get_card_with_keys(), cv2.COLOR_BGR2RGB))
#plt.show()

img_try=cv2.imread("images/Card2dist.jpeg")

img_bg = cv2.imread("images/Background.png")
bg_attr = background_attributes(img_bg)
thresh=50
count=1

d_M = mahalanobis(img_try, bg_attr)
img_seg = segment(img_try, d_M, thresh)
numlabels, img_blob, stats, centroids = get_blobs(img_seg, 3, 5)
blobs = []
for labels in range(1, numlabels):
    startpoint = (stats[labels, 0],stats[labels, 1])
    endpoint = (startpoint[0]+ stats[labels, 2], startpoint[1]+ stats[labels, 3]) 
    img_cropped = img_try[startpoint[1]:endpoint[1], startpoint[0]:endpoint[0], :]
    blob = Blobs(startpoint[1], startpoint[0], stats[labels, 2], stats[labels, 3], img_cropped)
    blobs.append(blob)

cards=[]

# FLANN Paramters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
#FLANN Creation
flann = cv2.FlannBasedMatcher(index_params, search_params)

descriptors_database=np.array([])
for card in deck:
    descriptors_database=np.append(descriptors_database, card.descriptors)

index_matches = []

for blob in blobs:
    keypoints, descriptors= key_and_desc(blob.image)
    matches = flann.knnMatch(descriptors, descriptors_database.reshape(-1,128).astype(np.float32), k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            #print(np.floor(m.trainIdx/100))
            index_matches.append(int(np.floor(m.trainIdx/100)))


index_count=np.zeros((len(deck),1))

for i in range(len(index_matches)):
    print(i)
    index_count[index_matches[i]]+=1

print(index_matches)
thresh_choice=5
index_count[index_count<thresh_choice]=0

for i in range (len(index_count)):
    if index_count[i]>0:
        plt.imshow(deck[i].image)
        plt.show()

#print(index_count)





    





