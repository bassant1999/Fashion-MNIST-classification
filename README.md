# Fashion-MNIST-classification
use CNN neural network on Fashion-MNIST dataset 
 # data set
 data set link: https://www.kaggle.com/datasets/zalando-research/fashionmnist
 
# steps
## Prepare the data
The dataset consists of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes (0 to 9).\
Labels: \
•	0 T-shirt/top \
•	1 Trouser \
•	2 Pullover \
•	3 Dress \
•	4 Coat \
•	5 Sandal \
•	6 Shirt \
•	7 Sneaker \
•	8 Bag \
•	9 Ankle boot 

a)	clean the data (remove the duplicates, and empty cells) \
b)	data visualization \
the train set: \
![Train set](images/train_set.PNG) \
The Test set: \
![Test set](images/test_set.PNG) \
Images from train set: \
![train set](images/images_from_train_set.PNG) \
Images from test set: \
![test set](images/images_from_test_set.PNG) 
 

The Distribution of digits in train set: \
 ![train set](images/distribution_of_digits.PNG) 

The Distribution of digits in test set: \
 ![train set](images/distribution_of_digits1.PNG) 
 
**Train dataset, and test dataset are balanced datasets.** 

c)	Encode the labels and reshaped the features (so that I can use them in CNN)


## I have used three techniques in training the model
### First technique:
a)	Splitting fashion-mnist_train dataset into train set, validation set. \
b)	Apply hyperparameter tunning on validation set. \
c)	Use the best hyperparameters from hyperparameters tunning step and use them to train the model on train set. \
The best hyperparameters: \
	'batch_size': 500, 'epochs': 30, 'model__activation': 'tanh', 'optimizer': 'Adam', 'optimizer__learning_rate': 0.001, 'optimizer__momentum': 0.0
#### The train accuracy vs epochs: 
![train set](images/train_acc_vs_epochs.PNG) \
**The train accuracy:** 92.23% 


d)	Evaluate on the test set (in fashion-mnist_test.csv) \
**The test Accuracy:** 89.579% 

### Second technique: 
a)	I have used 5-fold cross validation on fashion-mnist_train dataset. \
b)	Plot the accuracies vs epochs, and training loss vs epochs for each fold. 
#### The training accuracies vs epochs at different training folds:
![train set](images/acc_vs_epochs.PNG) 
 
### The validation loss vs Folds: 
![train set](images/val_loss_vs_epochs.PNG) 
### The training loss vs epochs at different training folds: 
![train set](images/loss_vs_epochs.PNG) 

c)	Get average training, and testing accuracies: \
	**Train average accuracy:** 88.167% \
	**Test average accuracy:** 87.25% 




### Third technique:
I have used transfer learning: \
a)	 I have used a pretrained VGG16 model on ImageNet, then use this model to train our fashion-mnist_train dataset through excluding the last 4 layers, and add new output layer, and train the weights of the output layer only while freezing the other weights, then evaluate resulted model on the test set (in fashion-mnist_test.csv). \
**The train accuracy:** 82.6% \
**The test accuracy:** 82.678% \
b)	I have used a pretrained VGG19 model on ImageNet, then use this model to train our fashion-mnist_train dataset through excluding the last 4 layers, and add new output layer, and train the weights of the output layer only while freezing the other weights, then evaluate resulted model on the test set (in fashion-mnist_test.csv). \
**The train accuracy:** 82.95% \
**The test accuracy:** 82.94% 


### The comparison of the three techniques:
![Results](images/compare.PNG) 

I got the best results from the first technique and second techniques, but the third technique was not the best because fashion mnist is slightly different from ImageNet. 





