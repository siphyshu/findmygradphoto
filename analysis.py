import os, cv2
from tqdm import tqdm
from collections import Counter

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join("Z:", "Convocation-Photos")

# there are 18 folders from 001 to 018 in the photos directory
# get the number of photos in each folder

total_photos = 0
for folder in os.listdir(PHOTOS_DIR):
    total_photos += len(os.listdir(os.path.join(PHOTOS_DIR, folder)))
print(f"Total photos: {total_photos}")

# stats on photo size and resolution
photo_sizes = []
photo_resolutions = []
for folder in os.listdir(PHOTOS_DIR):
    for photo in tqdm(os.listdir(os.path.join(PHOTOS_DIR, folder))):
        img = cv2.imread(os.path.join(PHOTOS_DIR, folder, photo))
        photo_sizes.append(img.shape[0] * img.shape[1])
        photo_resolutions.append((img.shape[0], img.shape[1]))

    print(f"Folder {folder} has {len(os.listdir(os.path.join(PHOTOS_DIR, folder)))} photos")
    print(f"Average photo size: {sum(photo_sizes) / len(photo_sizes)}")
    print("Most common photo resolution: ", Counter(photo_resolutions).most_common(1))
