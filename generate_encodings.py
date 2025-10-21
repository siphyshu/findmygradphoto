from tqdm import tqdm
import os, pickle, face_recognition

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join("Z:", "Convocation-Photos-Downscaled")
OUTPUT_FILE = os.path.join(ABSOLUTE_PATH, "face_encodings.pkl")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    print(f"📂 Loading images from: {PHOTOS_DIR}")
    
    # Get all image files
    image_files = [
        f for f in os.listdir(PHOTOS_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]
    
    print(f"🔍 Processing {len(image_files)} images...")
    
    all_encodings = {}
    skipped = 0
    errors = 0
    
    for filename in tqdm(image_files, desc="Generating encodings", unit="img"):
        path = os.path.join(PHOTOS_DIR, filename)
        
        try:
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)
            
            if not encodings:
                skipped += 1
                continue
            
            # Store all face encodings found in this image
            all_encodings[filename] = encodings
            
        except Exception as e:
            errors += 1
            print(f"\n❌ Error processing {filename}: {e}")
            continue
    
    # Save encodings to pickle file
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_encodings, f)
    
    print(f"\n✅ Successfully processed {len(all_encodings)} images with faces")
    print(f"⚠️  Skipped {skipped} images (no faces detected)")
    if errors > 0:
        print(f"❌ Errors: {errors} images failed")
    print(f"💾 Encodings saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()