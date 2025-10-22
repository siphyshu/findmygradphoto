# findmygradphoto 🎓

> "Why spend 20mins on a task manually, when you can spend 2hrs automating it?"

I live by one principle: if it _can_ be automated, it **must** be.

So when the college dumped thousands of graduation photos in a gdrive, naturally i wrote a script.

https://github.com/user-attachments/assets/53dcf9c4-e164-428f-927a-d367b3b93940


## What This Does

Three quick scripts that work together to find your face in a massive photo dump:

1. **downscale.py** - Makes processing faster by downscaling to 640px width
2. **generate_encodings.py** - Saves face embeddings in a pickle file to reuse later 
3. **find_face.py** - Matches your reference photo against all faces

Download the downscaled images from [this](https://drive.google.com/drive/folders/1JX7_5Om19WWfxFpyl0YUHD9HUlvro9GB?usp=sharing) gdrive (¬256MB), accesible only with VITB college ID.

## Installation

```bash
pip install opencv-python face_recognition tqdm numpy
```

**Note:** `face_recognition` needs cmake. The library tells you how to install it if you don't have it.

## How to Use

Can directly jump to step 3, since the encodings are included in the repo. 

If you do want to generate your own encodings, you can follow along.

### Step 1: Downscale Images

First, make the images smaller so it doesn't take forever to process 35GB of 4K photos:

```bash
python downscale.py
```

Edit the paths in the script to point to your photo folder. Uses multithreading because we're not savages.

### Step 2: Generate Face Encodings

Run face recognition on every photo and save the results:

```bash
python generate_encodings.py
```

This takes a while. Go grab coffee. It's turning every face into a 128-dimensional vector (fancy math stuff).

Saves everything to `face_encodings.pkl` so you don't have to do this again.

### Step 3: Find Your Face

The fun part - find yourself:

```bash
python find_face.py your_reference_photo.jpg
```

Use a clear photo of your face as reference. Script will:
- Compare your face against all the encodings
- Show top 10 matches in terminal
- Save ALL matches to `matches/` folder, ranked by similarity
- Let you view them one by one

**Optional: Adjust the threshold** if you're getting too many/few matches:

```bash
python find_face.py your_reference_photo.jpg 0.5
```

Lower number = stricter matching. Default is 0.6 which works pretty well.

## What's Inside

```
├── downscale.py              # Makes images smaller
├── generate_encodings.py     # Face recognition on everything
├── find_face.py              # Find yourself
├── face_encodings.pkl        # Generated database
└── matches/                  # Your photos (generated)
```

## How It Actually Works

1. **Downscale**: Resize photos to 640px width. Multithreaded because patience is not a virtue I possess.

2. **Encode**: Uses `face_recognition` to turn each face into a 128-dimensional vector. Math magic.

3. **Match**: Compares your reference face against all encodings using Euclidean distance. Lower distance = better match.

4. **Profit**: Get a ranked list of all photos with your face in them.

## Pro Tips

- Use a clear, front-facing photo as reference
- Preferably from the day of convocation in same attire
- Good lighting helps (obviously)
- Adjust threshold if results are weird
- The script skips photos without faces automatically

## Tech Stack

Python, OpenCV, face_recognition, tqdm for those satisfying progress bars

## License

MIT - do whatever you want with it

---

*Built for the 💖 of automation, and in celebration of graduating* 🎓
