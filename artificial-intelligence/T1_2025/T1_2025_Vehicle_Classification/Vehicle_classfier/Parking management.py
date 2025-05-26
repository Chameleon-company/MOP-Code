#!/usr/bin/env python
# coding: utf-8

# In[ ]:


dataset_dir = r"C:\Users\lokesh r\Documents\parking lot"  


# In[13]:


import os
import cv2
import numpy as np

img_size = 128 
labels = []
images = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))  
            img = img / 255.0  
            label = root.split("\\")[-1]  
            images.append(img)
            labels.append(label)

images = np.array(images)
labels = np.array(labels)

import matplotlib.pyplot as plt

plt.imshow(images[0])
plt.title(f"Label: {labels[0]}")
plt.show()


# In[3]:


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)

# Display the encoded labels for verification
print(labels_encoded[0])


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
print(f"Training data size: {len(X_train)} | Test data size: {len(X_test)}")


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[6]:


history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)


# In[7]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[8]:


def calculate_parking_fee(vehicle_type, duration):
    base_rate = 5  # Base rate in dollars per hour
    factor = {
        "small": 1.0,
        "medium": 1.5,
        "large": 2.0
    }
    return base_rate * duration * factor.get(vehicle_type.lower(), 1.0)

# Example usage:
vehicle_type = "medium"  
duration = 3.0  
fee = calculate_parking_fee(vehicle_type, duration)
print(f"Parking Fee: ${fee:.2f}")


# In[18]:


train_dir = r'C:\Users\lokesh r\Documents\parking lot\train' 
anno_train = r'C:\Users\lokesh r\Documents\parking lot\train\_annotations.coco.json' 

with open(anno_train, "r") as file:
    data = json.load(file)

for image_info in data['images']:
    image_id = image_info['id']
    image_path = os.path.join(train_dir, image_info['file_name'])

    image = cv2.imread(image_path)
    if image is None:
        continue

    for annotation in data['annotations']:
        if annotation['image_id'] == image_id: 
            category_id = annotation['category_id']
            bbox = annotation['bbox']
            x, y, w, h = map(int, bbox)

            if category_id == 1:
                color = (0, 255, 0)  
            elif category_id == 2:
                color = (0, 0, 255)  
            else:
                color = (255, 0, 0)  

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    break


# In[ ]:




