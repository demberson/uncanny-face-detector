import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from train_pca import load_data

# number of principle components (can be adjusted to change accuracy/creepiness of faces)
n_components = 20

def process_image(image_path, pca_model_path='models/pca_model.pkl'):
    # load pca model
    print(f"Loading model and applying filter with {n_components} components...")
    pca = joblib.load(pca_model_path)

    # get reference dimensions from dataset
    lfw_people = load_data()
    _, h, w = lfw_people.images.shape
    print(f"Required input size: {h}x{w}")

    # load user image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detect face
    detector = MTCNN()
    results = detector.detect_faces(img_rgb)

    if not results:
        print("No face detected.")
        return
    
    # take first face found
    x, y, width, height = results[0]['box']

    # extract face
    y1, y2 = max(0, y), min(img.shape[0], y + height)
    x1, x2 = max(0, x), min(img.shape[1], x + width)

    face_region = img_rgb[y1:y2, x1:x2]

    # pre-process for PCA
    face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    face_resized = cv2.resize(face_gray, (w, h))
    face_flat = face_resized.reshape(1, -1)

    # project face into eigenspace
    components = pca.transform(face_flat)

    # zero out all components after chosen number of components
    components_filtered = components.copy()
    components_filtered[:, n_components:] = 0

    # reconstruct faces w/ limited components
    reconstruction = pca.inverse_transform(components_filtered)
    reconstruction = reconstruction.reshape(h, w)

    # post-processing
    # resize image back to original size
    uncanny_face = cv2.resize(reconstruction, (x2-x1, y2-y1))

    # normalize pixel values
    uncanny_face = cv2.normalize(uncanny_face, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # convert back to color
    uncanny_face_color = cv2.cvtColor(uncanny_face, cv2.COLOR_GRAY2RGB)

    # masking
    final_image = img_rgb.copy()
    final_image[y1:y2, x1:x2] = uncanny_face_color

    # show results
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Uncanny Face (K={n_components})")
    plt.imshow(final_image)
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    process_image("selfie.jpg")
