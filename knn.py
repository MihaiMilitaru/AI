# Importing the libraries
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score

def normalizare(train, validation):
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train = train-mean
    train = train/std
    validation = validation-mean
    validation = validation/std

    return train, validation

# citirea si incarcarea datelor de train
f=open('train_labels.txt','r')
train_labels=f.readlines()[1:15001] # skip the first line (header)

train_images_names = []
train_images_labels = []

# citim primele 15000 de imagini si le salvam in train_images_names
# citim primele 15000 de etichete si le salvam in train_images_labels
for i in range(len(train_labels)):
    train_images_names.append(train_labels[i].split(',')[0])
    train_images_labels.append(train_labels[i].split(',')[1])

train_images_normalized = []
train_labels = []

# citim imaginile si le salvam in train_images_normalized
for i in range(len(train_images_names)):
    image = np.asarray(Image.open('data/'+train_images_names[i]+'.png').convert('L').getdata())
    train_images_normalized.append(image)
    train_labels.append(train_images_labels[i])

# transformam listele in array-uri numpy
train_images_normalized = np.array(train_images_normalized)
train_labels = np.array(train_labels)


# citirea si incarcarea datelor de validare
f=open('train_labels.txt','r')
validation_labels=f.readlines()[15001:17001] # read validation labels

validation_images_names = []
validation_images_labels = []

# citim primele 2000 de imagini si le salvam in validation_images_names
# citim primele 2000 de etichete si le salvam in validation_images_labels
for i in range(len(validation_labels)):
    validation_images_names.append(validation_labels[i].split(',')[0])
    validation_images_labels.append(validation_labels[i].split(',')[1])

validation_images_normalized = []
validation_labels = []


# citim imaginile si le salvam in validation_images_normalized
for i in range(len(validation_images_names)):
    image = np.asarray(Image.open('data/'+validation_images_names[i]+'.png').convert('L').getdata())
    validation_images_normalized.append(image)
    validation_labels.append(validation_images_labels[i])

# transformam listele in array-uri numpy
validation_images_normalized = np.array(validation_images_normalized)
validation_labels = np.array(validation_labels)


# citirea si incarcarea datelor de testare
test_images_normalized = []


for i in range(17001,22150):
    image = np.asarray(Image.open('data/'+'0'+str(i)+'.png').convert('L').getdata())
    test_images_normalized.append(image)

test_images_normalized = np.array(test_images_normalized)

# transformam datele in float64
test_images_normalized = test_images_normalized.astype('float64')
train_images_normalized = train_images_normalized.astype('float64')
validation_images_normalized = validation_images_normalized.astype('float64')

# normalizarea datelor
train_images_normalized, validation_images_normalized = normalizare(train_images_normalized, validation_images_normalized)


# KNN
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_images_normalized, train_labels)
predictions = classifier.predict(validation_images_normalized)
validation = f1_score(validation_labels, predictions, average='macro')
acc_score = accuracy_score(validation_labels, predictions, normalize=True)
recall = recall_score(validation_labels, predictions, average='macro')

print("KNN F1 Score : " + str(validation))
print("KNN Accuracy Score : " + str(acc_score))
print("KNN Recall Score : " + str(recall))

# matricea de confuzie

cm = confusion_matrix(validation_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.show()


# fisierele de output in care se vor salva rezultatele sub forma de id si clasa

g=open('sample_submission.csv','w')
g.write('id,class\n')
for i in range(len(predictions)):
    index = i + 17001
    g.write(str(index).zfill(6) + ',' + str(predictions[i]) )
