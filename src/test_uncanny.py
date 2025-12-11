import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from train_pca import load_data

# number of principle components (can be adjusted to change accuracy/creepiness of faces)
n_components = 80

BLUR_AMOUNT = 5
NOISE_LEVEL = 15

def add_grain(image, intensity=10):
    h, w = image.shape
    noise = np.random.normal(0, intensity, (h, w)).astype('uint8')
    grainy = cv2.add(image, noise)
    return grainy

def match_brightness(source, reference):
    # calculate mean and standard deviation
    src_mean, src_std = cv2.meanStdDev(source)
    ref_mean, ref_std = cv2.meanStdDev(reference)

    # linear transformation
    adjustment = (source.astype('float32') - src_mean) * (ref_std / src_std) + ref_mean

    return np.clip(adjustment, 0, 255).astype('uint8')

def process_image(image_path, pca_model_path='models/pca_model.pkl'):
    # load pca model
    print(f"Loading model and applying filter with {n_components} components...")
    pca = joblib.load(pca_model_path)

    # get reference dimensions from dataset
    lfw_people = load_data()
    _, h, w = lfw_people.images.shape

    # load user image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image.")
        return
    
    # make image grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect face
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    face_region = img_gray[y1:y2, x1:x2]

    # pre-process for PCA
    face_resized = cv2.resize(face_region, (w, h))
    face_flat = face_resized.reshape(1, -1)

    # project face into eigenspace
    components = pca.transform(face_flat)

    # zero out all components after chosen number of components
    components[:, n_components:] = 0

    # reconstruct faces w/ limited components
    reconstruction = pca.inverse_transform(components)
    reconstruction = reconstruction.reshape(h, w)

    # post-processing
    # resize image back to original size
    uncanny_face = cv2.resize(reconstruction, (x2-x1, y2-y1))
    uncanny_face = match_brightness(uncanny_face, face_region)
    uncanny_face = cv2.normalize(uncanny_face, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # masking
    mask = np.zeros_like(uncanny_face)
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    axes = (mask.shape[1] // 2, mask.shape[0] // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255), -1)

    # blur
    mask = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0

    # create final image
    final_face = (uncanny_face * mask + face_region * (1.0 - mask)).astype('uint8')
    final_image = img_gray.copy()
    final_image = cv2.GaussianBlur(final_image, (BLUR_AMOUNT, BLUR_AMOUNT), 0)
    final_image[y1:y2, x1:x2] = final_face
    final_image = add_grain(final_image, intensity=NOISE_LEVEL)

    # show result
    plt.figure(figsize=(10,5))
    plt.imshow(final_image, cmap='gray')
    plt.title(f"Uncanny Face (K={n_components})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    process_image("test_selfie.jpg")
