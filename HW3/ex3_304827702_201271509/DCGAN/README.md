## GAN

The main parts:

* Data preprocessing- loading the data, renormalizing to range [-1,1], and split to training and test set
* DCGAN classes - GeneratorDCGAN to generate image from random (uniform) noise, and Discriminator that should distinguish between real and fake images
* Hyperparameter generator- creating different configs for different runs
* Modelrun- a class to create/load a model with a specific configuration
* RunConfig- a class to fit, evaluate and save a model
* plot section- plot generator and discriminator loss with tensorboard, and plot images and loss to file (matplotlib)
* main - a function to start run(s)
* Save real imges- choose a list of classes as ground-truth and save the corresponding real images examples from test set (unobserved)
* Test pre-trained model- choose the relevant indexes from the fixed noise to compare (plots) between pre-trained fake images and real images


### Run pre-trained model:
Run all the cells above until you reach to main and skip to 'Save real imges (for plot)' 'Test pre-trained model' section to plot

### Run a model from scratch:
Run all the cells above until you reach 'Test pre-trained model' section


Note:
* The training process and plots are similar to as in https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html