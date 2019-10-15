from emnist import extract_training_samples
from emnist import extract_test_samples
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

def makes_one_hot_vectors(array):
    temp = array
    one_hot_vector = np.zeros((temp.size,temp.max()+1))
    one_hot_vector[np.arange(temp.size),temp] = 1
    return one_hot_vector

    # rescale np.array from uint8 to float32
def rescale_to_float32(array):
    return array.astype(np.float32) / 255

    # Normalize images
def normalize_image(image):
    flip_x = np.fliplr(image).copy()
    rotate90_aCW = np.rot90(flip_x).copy()
    normalized_image = rotate90_aCW.copy()
    return normalized_image

class EMNIST_Letters:
    def __init__(self, train_images, train_labels, test_images, test_labels,  batch):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        
        self.current_epoch = 0
        
        # divided the training dataset into 'batch' chunks
        self.batched_train_dataset = np.array_split(self.train_images, len(self.train_images)/batch)
        self.batched_test_dataset = np.array_split(self.test_images, len(self.test_images)/batch)

    def train(self):
        return self.train_images, self.train_labels

    def test(self):
        return self.test_images, self.test_labels
        
    def set_current_epoch(self, new_epoch):
        self.current_epoch += new_epoch
        
    def get_current_batch(self):
        return self.current_epoch
    
    def get_data_shapes(self):
        print('train_images: ', self.train_images.shape)
        print('train_labels: ', self.train_labels.shape)

    def get_batched_train(self):
        return self.batched_train_dataset

    def get_batched_test(self):
        return self.batched_test_dataset
 
def emnist_factory():
    train_images, train_labels = extract_training_samples('letters')
    test_images, test_labels = extract_test_samples('letters')
    one_hot_train_labels = makes_one_hot_vectors(train_labels)
    one_hot_test_labels = makes_one_hot_vectors(test_labels)
    batch = 128
    return EMNIST_Letters(train_images, one_hot_train_labels,test_images, one_hot_test_labels,batch)

if __name__ == '__main__':
    
    emnist = emnist_factory()
    
    
    
    

##    plt.figure(figsize=(1,1))
##    sample_image = train_images[0]
##    sample_image = sample_image.reshape([28,28])
##
##
##    plt.imshow(rescale_to_float32(sample_image), cmap='Greys')
##    plt.show()
