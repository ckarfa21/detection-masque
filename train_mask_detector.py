# utilisation
# python train_mask_detector.py --dataset dataset

# import des bibliotheques
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# constructeur d'arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# initialiser le taux d'apprentissage initial, le nombre d'épochs  pour lesquelles il faut se former,
# et taille du lot
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# récupérer la liste des images dans notre répertoire d'ensembles de données, puis initialiser
# la liste des données (c'est-à-dire des images) et des images de classe
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# boucle sur les chemins de l'image
for imagePath in imagePaths:
	# extraire le label de classe du nom du fichier
	label = imagePath.split(os.path.sep)[-2]

	# charger l'image d'entrée (224x224) et la prétraiter
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# mettre à jour les listes de données et les labels, respectivement
	data.append(image)
	labels.append(label)

#convertir les données et les labels en tableaux NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# effectuer un encodage en une seule fois sur les labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# diviser les données en fractionnements d'apprentissages et de test en utilisant 75 % de
# les données pour l'apprentissage et les 25% restants pour les tests
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

#construire le générateur d'images pour l'apprentissage et l'augmentation des données
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# charger le réseau MobileNetV2, en s'assurant que les couches FC principales sont
# laissé de côté
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construire la tête du modèle qui sera placé sur le
# le modèle de base
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# boucler sur toutes les couches du modèle de base et les figer pour qu'elles
# ne sois pas  mises à jour lors du premier processus de formation
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compiler notre modele 
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# former la tete du réseau
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

#faire des prévisions sur le banc d'essai
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# pour chaque image de la série de tests, nous devons trouver l'index de 
#  chaque étiquette avec la plus grande probabilité prédite correspondante
predIdxs = np.argmax(predIdxs, axis=1)

# montrer un rapport de classification bien formaté
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# sérialiser le modèle sur disque
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# tracer la perte de formation et la précision
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])