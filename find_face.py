import os, pickle, face_recognition, cv2
import numpy as np
from tqdm import tqdm

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.path.join(ABSOLUTE_PATH, "face_encodings.pkl")
PHOTOS_DIR = os.path.join("Z:", "Convocation-Photos-Downscaled")
RESULTS_DIR = os.path.join(ABSOLUTE_PATH, "matches")

# Threshold for face matching (lower = stricter, default is 0.6)
MATCH_THRESHOLD = 0.6


def load_encodings():
    """Load face encodings from pickle file."""
    print(f"📂 Loading encodings from: {ENCODINGS_FILE}")
    with open(ENCODINGS_FILE, "rb") as f:
        encodings = pickle.load(f)
    print(f"✅ Loaded encodings for {len(encodings)} images")
    return encodings


def get_reference_face(reference_image_path):
    """Extract face encoding from reference image."""
    print(f"🔍 Processing reference image: {reference_image_path}")
    
    img = face_recognition.load_image_file(reference_image_path)
    encodings = face_recognition.face_encodings(img)
    
    if not encodings:
        raise ValueError("❌ No face detected in reference image!")
    
    if len(encodings) > 1:
        print(f"⚠️  Multiple faces detected ({len(encodings)}), using the first one")
    
    print("✅ Reference face encoding generated")
    return encodings[0]


def find_matches(reference_encoding, all_encodings, threshold=MATCH_THRESHOLD):
    """Find all matching faces in the encoding database."""
    print(f"\n🔎 Searching for matches (threshold: {threshold})...")
    
    matches = []
    
    for filename, face_encodings in tqdm(all_encodings.items(), desc="Comparing faces", unit="img"):
        # Compare reference face with all faces in this image
        for face_idx, encoding in enumerate(face_encodings):
            distance = face_recognition.face_distance([encoding], reference_encoding)[0]
            
            if distance <= threshold:
                matches.append({
                    'filename': filename,
                    'face_index': face_idx,
                    'distance': distance,
                    'similarity': (1 - distance) * 100  # Convert to percentage
                })
    
    # Sort by distance (best matches first)
    matches.sort(key=lambda x: x['distance'])
    
    return matches


def display_matches(matches):
    """Display matching images."""
    if not matches:
        print("\n😞 No matches found. Try increasing the threshold or use a clearer reference image.")
        return
    
    print(f"\n🎉 Found {len(matches)} matches!")
    print("\nTop matches:")
    
    for i, match in enumerate(matches[:10], 1):  # Show top 10
        print(f"{i}. {match['filename']} - Similarity: {match['similarity']:.1f}% (distance: {match['distance']:.4f})")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Copy matches to results folder
    print(f"\n💾 Saving matches to: {RESULTS_DIR}")
    for i, match in enumerate(matches, 1):
        src_path = os.path.join(PHOTOS_DIR, match['filename'])
        
        # Add rank prefix to filename
        dst_filename = f"{i:03d}_{match['similarity']:.0f}pct_{match['filename']}"
        dst_path = os.path.join(RESULTS_DIR, dst_filename)
        
        # Read and save image (with face box drawn optionally)
        img = cv2.imread(src_path)
        if img is not None:
            cv2.imwrite(dst_path, img)
    
    print(f"✅ Saved {len(matches)} matching images to {RESULTS_DIR}")
    
    # Ask if user wants to view matches
    print("\n📸 Press any key to view next match, ESC to exit...")
    for match in matches:
        img_path = os.path.join(PHOTOS_DIR, match['filename'])
        img = cv2.imread(img_path)
        
        if img is not None:
            # Add text overlay with similarity score
            text = f"Similarity: {match['similarity']:.1f}% - {match['filename']}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow("Match Found!", img)
            key = cv2.waitKey(0)
            
            if key == 27:  # ESC key
                break
    
    cv2.destroyAllWindows()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python find_face.py <reference_image_path> [threshold]")
        print("Example: python find_face.py my_photo.jpg 0.6")
        print("\nThreshold (optional): Lower = stricter matching (default: 0.6)")
        sys.exit(1)
    
    reference_image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else MATCH_THRESHOLD
    
    if not os.path.exists(reference_image_path):
        print(f"❌ Reference image not found: {reference_image_path}")
        sys.exit(1)
    
    if not os.path.exists(ENCODINGS_FILE):
        print(f"❌ Encodings file not found: {ENCODINGS_FILE}")
        print("Please run generate_encodings.py first!")
        sys.exit(1)
    
    # Load saved encodings
    all_encodings = load_encodings()
    
    # Get reference face encoding
    reference_encoding = get_reference_face(reference_image_path)
    
    # Find matches
    matches = find_matches(reference_encoding, all_encodings, threshold)
    
    # Display results
    display_matches(matches)


if __name__ == "__main__":
    main()

