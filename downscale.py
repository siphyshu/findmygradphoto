import os, cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

INPUT_DIR = os.path.join("Z:", "Convocation-Photos")
OUTPUT_DIR = os.path.join("Z:", "Convocation-Photos-Downscaled")

MAX_WIDTH = 640
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
NUM_THREADS = multiprocessing.cpu_count() * 2  # Use 2x CPU cores for I/O-bound tasks

def downscale_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    if w <= MAX_WIDTH:  # no need to downscale
        cv2.imwrite(output_path, img)
        return True

    scale = MAX_WIDTH / w
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)
    return True


def process_single_image(args):
    """Worker function to process a single image."""
    in_path, out_path = args
    try:
        downscale_image(in_path, out_path)
        return True
    except Exception as e:
        print(f"\n❌ Error processing {in_path}: {e}")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all image paths (save all to single output folder)
    image_pairs = []
    for root, dirs, files in os.walk(INPUT_DIR):
        rel_path = os.path.relpath(root, INPUT_DIR)
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            
            in_path = os.path.join(root, file)
            
            # Create unique filename using subfolder prefix
            if rel_path != ".":
                subfolder = rel_path.replace(os.sep, "_")
                out_filename = f"{subfolder}_{file}"
            else:
                out_filename = file
            
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            if os.path.exists(out_path):
                continue  # already processed

            image_pairs.append((in_path, out_path))

    print(f"🚀 Processing {len(image_pairs)} images using {NUM_THREADS} threads...")

    # Process images in parallel with progress bar
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        list(tqdm(
            executor.map(process_single_image, image_pairs),
            total=len(image_pairs),
            desc="📸 Processing images",
            unit="img"
        ))

    print("✅ All images downscaled and saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
