Deep Learning Flower Identifier
PyTorch implementation of a Deep learning network to identify 102 different types of flowers (developed for the PyTorch Scholarship Challenge).

The used data set contains images of flowers from 102 different species divided in a training set and a validation set.The images can be downloaded here: https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip

In addition, the repository contains a utility for testing the performance of a model on the original flower dataset (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) 

This allows to evaluate if the model is not affected by overfitting and if it is suitable for a real usecase.

-Train the model
-Test the model
-Publish the model results
-Publish own model rrsults
-Test set

Train Model:
To train the model, I ultilized the environment beklow:

Google Colaboratory

Tset the model:

To test a saved model, simply load it into memory and recall the function calc_accuracy:

from test_model_pytorch_facebook_challenge import calc_accuracy

model = load_model('classifier.pth')
calc_accuracy(model, input_image_size=224)
If you are using a notebook

!git clone https://github.com/My-Roadmap-to-Pytorch-DL-Nanodegree/deep-learning-flower-identifier
!pip install requests
!pip install airtable
import sys
sys.path.insert(0, 'deep-learning-flower-identifier')
from test_model_pytorch_facebook_challenge import calc_accuracy

model = load_your_model('classifier.pth')
calc_accuracy(model, input_image_size=224)
The accuracy will be calculated on a dataset, downloaded on runtime, containing images classified by category, downloaded from Google search results.

You can also use the function to calculate the model's performance on a custom directory.

usage:
calc_accuracy(model, input_image_size, use_google_testset=False, testset_path=None, batch_size=32,
              norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]))                        
"""
Calculate the mean accuracy of the model on the test test
:param model: the model
:param use_google_testset: If true use the testset derived from google image
:param testset_path: custom test set or if default google images dataset is downloaded
:param batch_size:
:param input_image_size:
:param norm_mean: normalizazition mean for RGB channel
:param norm_std: stardard deviation mean for RGB channel
:return: the mean accuracy
"""
Publish the model results:
For collaborative purposes you can publish the results of your model online, along with a series of (optional) information on your model's structure.

The public results are visible here: https://airtable.com/shrCs1LDFdBus0tMG

To publish the results you can simply call the method publish_evaluated_model

from test_model_pytorch_facebook_challenge import publish_evaluated_model
publish_evaluated_model(model, input_image_size=224,  username="@Slack.Username", model_name="VGG19", optim="Adam",
                        criteria="NLLLoss", scheduler="StepLR", epoch=10)

Publish own model results:
From colab
Add the code cell below:

!git clone https://github.com/My-Roadmap-to-Pytorch-DL-Nanodegree/deep-learning-flower-identifier
!pip install requests
!pip install airtable
import sys
sys.path.insert(0, 'deep-learning-flower-identifier')

from test_model_pytorch_facebook_challenge import publish_evaluated_model

model = load_your_model('classifier.pth')

publish_evaluated_model(model, input_image_size=224,  username="@Slack.Username", model_name="VGG19", optim="Adam",
                    criteria="NLLLoss", scheduler="StepLR", epoch=10)

from test_model_pytorch_facebook_challenge import publish_evaluated_model
publish_evaluated_model(model, input_image_size=224,  username="@Slack.Username", model_name="VGG19", optim="Adam",
                    criteria="NLLLoss", scheduler="StepLR", epoch=10)

Test Set:
If you are just interested in the test set, I downloaded them here:

Original test set: https://www.dropbox.com/s/da6ye9genbsdzbq/flower_data_original_test.zip?dl=1

Google test set: https://www.dropbox.com/s/3zmf1kq58o909rq/google_test_data.zip?dl=1

If you are using a notebook you can use it directly:

!wget -O flower_data_orginal_test.zip "https://www.dropbox.com/s/da6ye9genbsdzbq/flower_data_original_test.zip?dl=1"
!unzip flower_data_orginal_test.zip


!wget -O google_test_data.zip "https://www.dropbox.com/s/3zmf1kq58o909rq/google_test_data.zip?dl=1"
!unzip google_test_data.zip
