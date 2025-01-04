import cv2
from .utils import Dataset, BoW, ImageClassifier
import time
from tqdm import tqdm
import sys
import pickle

# Cargar conjuntos de datos
training_set = Dataset.load("data/traffic_Data/DATA", "*png")
validation_set = Dataset.load("data/traffic_Data/TEST", "*png")

# Define constants
vocabulary_size = 200
iterations = 40
termination_criteria = (
    cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
    iterations,
    1e-6,
)


print(training_set[0])
print(validation_set[0])

# Crear el extractor de características SIFT
feature_extractor = cv2.SIFT_create()

# Extraer descriptores
print("\nComputing SIFT descriptors...")
time.sleep(0.1)  # Previene problemas de concurrencia entre tqdm y print

descriptors = []  # Lista para almacenar descriptores
for path in tqdm(training_set, unit="image", file=sys.stdout):
    # Cargar la imagen en escala de grises
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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

try:
    with open(filename, "rb") as f:
        data = pickle.load(f)
        print(f"Data from vocab.pickle: {data}")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")


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
