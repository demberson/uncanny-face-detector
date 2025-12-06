import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

def load_data():

    # fetch LFW dataset

    print("Loading LFW dataset...")

    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4) # downscaling images to help with the uncanny effect

    n_samples, h, w = lfw_people.images.shape

    print(f"Dataset loaded")
    print(f"Total samples: {n_samples}")
    print(f"Image dimensions: {h}x{w} pixels")

    return lfw_people

if __name__ == "__main__":
    data = load_data()

    plt.imshow(data.images[0], cmap='gray')
    plt.title(f"First sample: {data.target_names[data.target[0]]}")
    plt.show()