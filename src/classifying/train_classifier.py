import cv2
from .utils import Dataset, BoW, ImageClassifier
import time
from tqdm import tqdm
import sys
import pickle
import numpy as np

# Cargar conjuntos de datos
training_set = Dataset.load("data/traffic_Data/DATA", "*png")
validation_set = Dataset.load("data/traffic_Data/TEST", "*png")

print(training_set[0])
print(validation_set[0])

# Define constants
vocabulary_size = 1024
iterations = 40
termination_criteria = (
    cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
    iterations,
    1e-6,
)

# Crear el extractor de características SIFT
feature_extractor = cv2.SIFT_create()

# Extraer descriptores
print("\nComputing SIFT descriptors...")
time.sleep(0.1)  # Previene problemas de concurrencia entre tqdm y print

descriptors = []  # Lista para almacenar descriptores
for path in tqdm(training_set, unit="image", file=sys.stdout):
    # Cargar la imagen en escala de grises
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Limpieza de la imagen
    # 1. Ecualización del histograma
    image_eq = cv2.equalizeHist(image)

    # 2. Filtrado de ruido
    image_blur = cv2.GaussianBlur(image_eq, (5, 5), 0)

    # 3. Ajustar brillo
    image_bright = cv2.convertScaleAbs(image_blur, alpha=1.0, beta=50)

    # 4. Corrección gamma
    gamma = 2.0  # Ajustar el valor de gamma según sea necesario
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype="uint8")

    image = gamma_corrected
    try:
        # Detectar y describir características SIFT
        keypoints, descriptor = feature_extractor.detectAndCompute(image, None)
        if descriptor is not None:
            descriptors.append(descriptor)
        else:
            print(f"Advertencia: No se encontraron descriptores en {path}")
    except Exception as e:
        print(f"Error procesando {path}: {e}")
print(f"Número de descriptores extraídos: {len(descriptors)}")

# Initialize BOWKMeansTrainer
words = cv2.BOWKMeansTrainer(vocabulary_size, termination_criteria)

# Add descriptors to the trainer
for desc in tqdm(descriptors, desc="Adding descriptors"):
    words.add(desc)

time.sleep(0.1)  # Prevent tqdm printing issues
print("\nClustering descriptors into", vocabulary_size, "words using K-means...")

# Perform k-means clustering to build the vocabulary
vocabulary = words.cluster()
print(f"Vocab: {vocabulary}")
filename = "classifying/vocabulary.pickle"
# Open the file from above in the write and binay mode
with open(filename, "wb") as f:
    pickle.dump(["SIFT", vocabulary], f, pickle.HIGHEST_PROTOCOL)


# Load vocabulary into BoW
bow = BoW()
bow.load_vocabulary(filename.replace(".pickle", ""))

# Train the image classifier
image_classifier = ImageClassifier(bow)
print(f"Image classifier: {image_classifier}")
image_classifier.train(training_set, iterations)

# Save the trained classifier
classifier = "classifying/classifier"
image_classifier.save(classifier)

print("Vocabulary and classifier successfully built and saved.")


print("Empezando evaluación en el train: ")

bow = BoW()
# Especify the args for the loading method
bow.load_vocabulary(filename.replace(".pickle", ""))

image_classifier = ImageClassifier(bow)
# Especify the args for the loading method
image_classifier.load(classifier)
# Especify the args for the loading method
accuracy, confusion_matrix, classification = image_classifier.predict(training_set)

print("Empezando evaluación en el test: ")
bow = BoW()
# Especify the args for the loading method
bow.load_vocabulary(filename.replace(".pickle", ""))

image_classifier = ImageClassifier(bow)
# Especify the args for the loading method
image_classifier.load(classifier)
# Especify the args for the loading method
accuracy, confusion_matrix, classification = image_classifier.predict(validation_set)
