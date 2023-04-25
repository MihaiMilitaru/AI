# Importing the libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, recall_score

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing.image import ImageDataGenerator



# read training and validation labels
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', 'r') as f:
    train_labels = np.array([line.strip().split(',') for line in f.readlines()[1:]])

with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', 'r') as g:
    validation_labels = np.array([line.strip().split(',') for line in g.readlines()[1:]])

# extract image names and load images
train_images_normalized = np.array([cv2.normalize(cv2.cvtColor(cv2.imread('/kaggle/input/unibuc-brain-ad/data/data/' + name + '.png').astype(np.float64), cv2.COLOR_BGR2GRAY), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for name in [label[0] for label in train_labels]])
validation_images_normalized = np.array([cv2.normalize(cv2.cvtColor(cv2.imread('/kaggle/input/unibuc-brain-ad/data/data/' + name + '.png').astype(np.float64), cv2.COLOR_BGR2GRAY), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for name in [label[0] for label in validation_labels]])

# load and convert test images to grayscale
test_images_normalized = np.array([cv2.normalize(cv2.cvtColor(cv2.imread('/kaggle/input/unibuc-brain-ad/data/data/0' + str(name) + '.png').astype(np.float64), cv2.COLOR_BGR2GRAY), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for name in range(17001, 22150)])

# resize images to 128x128
train_images_normalized = np.array([cv2.resize(img, (128, 128)) for img in train_images_normalized])
validation_images_normalized = np.array([cv2.resize(img, (128, 128)) for img in validation_images_normalized])
test_images_normalized = np.array([cv2.resize(img, (128, 128)) for img in test_images_normalized])


# print(train_images_normalized.shape)
# print(validation_images_normalized.shape)
# print(test_images_normalized.shape)

# convert arrays to numpy arrays
train_labels=np.array([int(label[1]) for label in train_labels])
validation_labels=np.array([int(label[1]) for label in validation_labels])

nr_train_samples = 15000
nr_validation_samples = 2000

# data augmentation
# flip horizontally and vertically
train_datagen = ImageDataGenerator(

    horizontal_flip=True,
    vertical_flip=True,

)

# create a generator
# add a dimension to the images beacuse the model expects a 4D array
train_generator = train_datagen.flow(
    np.expand_dims(train_images_normalized, axis=-1), train_labels,  # training data and labels
    batch_size=32,  # batch size for training
)

classifier = Sequential([

    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(1, activation='sigmoid')

])


classifier.summary()

classifier.compile(loss=BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
history = classifier.fit(train_images_normalized, train_labels, epochs=75, validation_data=(validation_images_normalized, validation_labels),
                         batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=9, restore_best_weights=True)],
                         validation_steps=nr_validation_samples / 32, steps_per_epoch=nr_train_samples // 32)

validationLoss, validationAccuracy = classifier.evaluate(validation_images_normalized, validation_labels)

print('Validation accuracy:', validationAccuracy)

print('Validation loss:', validationLoss)

# plot the evolution of the train and validation accuracy

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)
plt.plot(epochs, train_acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# predictions

predictions = classifier.predict(test_images_normalized)
predictions = np.round(predictions)


predictions_validation = classifier.predict(validation_images_normalized)
predictions_validation = np.round(predictions_validation)
f1_score = f1_score(validation_labels, predictions_validation, average='binary')
acc_score = accuracy_score(validation_labels, predictions_validation, normalize=True)
recall_score = recall_score(validation_labels, predictions_validation, average='binary')

print('F1 score:', f1_score)
print('Accuracy score:', acc_score)
print('Recall score:', recall_score)

# confusion matrix

cm = confusion_matrix(validation_labels, predictions_validation)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.show()



with open('sample_submission.csv', 'w') as f:
    f.write('id,class\n')
    for i, pred in enumerate(predictions):
        index = i + 17001
        f.write(f"{index:06d},{int(pred)}\n")