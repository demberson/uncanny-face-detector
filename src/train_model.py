import os
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from train_pca import load_data

def train_and_save():
    # load data
    lfw_people = load_data()
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data

    # compute mean face and eigenfaces
    print(f"Extracting top 150 eigenfaces from {n_samples} faces...")
    pca = PCA(n_components=150, whiten=True).fit(X)

    # save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pca, 'models/pca_model.pkl')
    print("Model saved to models/pca_model.pkl")

    # reshape 1D arrays back to 2D images
    eigenfaces = pca.components_.reshape((150, h, w))

    # show mean face
    plt.figure(figsize=(5, 5))
    plt.imshow(pca.mean_.reshape(h, w), cmap='gray')
    plt.title("The Mean Face")
    plt.axis('off')

    # show top 12 eigenfaces
    plt.figure(figsize=(10, 5))
    plt.suptitle("The First 12 Eigenfaces", fontsize=16)
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(eigenfaces[i], cmap='gray')
        plt.title(f"Eigenface {i+1}")
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    train_and_save()
