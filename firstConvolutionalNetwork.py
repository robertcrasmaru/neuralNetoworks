from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

#import the data

trainDataGen = ImageDataGenerator(
 rescale = 1./255,
 shear_range = 0.2, 
 zoom_range = 0.2, 
 horizontal_flip = True )

testDataGen = ImageDataGenerator( rescale = 1./255 )
training_set = trainDataGen.flow_from_directory( 
	'dataSet/trainingSet/',
	target_size = (64, 64),
	batch_size = 32,
	class_mode = 'binary'
	)
test_set = testDataGen.flow_from_directory( 
	'dataSet/testSet/dogs/',
	target_size = (64, 64),
	batch_size = 32,
	class_mode = 'binary'
	)

#create the CNN

myModel = Sequential()

#layer1
myModel.add(
	Conv2D(32 , (3,3), input_shape = (64, 64, 3), activation = 'relu'))
myModel.add(
	MaxPooling2D( pool_size = (2, 2)))
#layer2
myModel.add(
	Conv2D(32 , (3,3), input_shape = (64, 64, 3), activation = 'relu'))
myModel.add(
	MaxPooling2D( pool_size = (2, 2)))
#layer3
myModel.add(
	Conv2D(64 , (3,3), input_shape = (64, 64, 3), activation = 'relu'))
myModel.add(
	MaxPooling2D( pool_size = (2, 2)))
#flattening layer
myModel.add(Flatten())
#fully connected layer
myModel.add(Dense(units = 128, activation = 'relu'))
myModel.add(Dense(units = 1, activation = 'sigmoid'))

myModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
	metrics = ['accuracy'])


#train the CNN

myModel.fit_generator(
	training_set,
	steps_per_epoch = 8000,
	epochs = 25,
	validation_data = test_set,
	validation_steps = 1500)


#make prediction

testImage1 = image.load_img('dataSet/dog.jpg', target_size = (64, 64))
testImage2 = image.load_img('dataSet/wookie.jpg', target_size = (64, 64))

testImage1 = image.img_to_array(testImage1)
testImage2 = image.img_to_array(testImage2)

testImage1 = np.expand_dims(testImage1, axis = 0)
testImage2 = np.expand_dims(testImage2, axis = 0)

result = myModel.predict(testImage1)

training_set.class_indices
if result[0][0] == 1:
	prediction = 'yes'
else:
	prediction = 'no'

print(prediction)