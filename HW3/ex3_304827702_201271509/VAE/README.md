## VAE

The main parts:

* Data preprocessing- loading the data, and split to training, validation and test
* VAE class - use VAE to encdoe the images
* Hyperparameter generator- creating different configs for different runs
* Modelrun- a class to create/load a model with a specific configuration
* RunConfig- a class to fit, evaluate and save a model
* plot section- plot validation with tensorboard, and plot test to file (matplotlib)
* Train VAE - a "wrapper" function to train VAE
* DatasetCV - a class to generate a fixed #training examples (after dimensionality reduction) per label (a total of {100, 600, 1000, 3000} examples)
and splitting the dataset to apply CV (k=10)
* TSVC - a class to create/load SVC for classification
* Train TSVM on latent examples- hyperparameter tuning and save best SVC models per k
* Test pre-trained model- run a pretrained model
* Load (latent) test set and best TSVM models, and calculate total number of errors

### Run pre-trained model:
Run all the cells above until you reach to train_vae(), and skip to
the 'Load (latent) test set and best TSVM models, and calculate total number of errors' section (run all cells from 'Run on test set with best hyperparameter for each regularization' section)


### Run a model from scratch:
* VAE- run all the cells above until you reach to the cell below Find best hyperparameters using validation set
* TSVM- after choosing (and saving best model) run the Tranductive SVM section (until you reach load test set..) to train best models 
  
Notes:
* The network architecture is similar to as implemented in https://github.com/pytorch/examples/blob/master/vae/main.py