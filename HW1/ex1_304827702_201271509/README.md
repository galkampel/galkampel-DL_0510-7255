
## The main parts:
* Loading the data
* LeNet5 class
* Hyperparameter generator- creating different configs. for different runs
* Modelrun- a class to create/load a model with a specific configuration
* RunConfig- a class to fit, evaluate and save a model
* plot section- plot validation with tensorboard, and plot test to file (matplotlib)
* main- a function to start run(s)
* Test pre-trained model- run a pretrained model

## Run pre-trained model:
Run all the cells above until you reach to main(), and skip to
the 'Test pre-trained model' section (run all cells in that section)


## Run a model from scratch:
In order to run main you need to run all the cells above.
* Use 'Find best hyperparameters using validation set for each regularization'
cells to find best hyperparams (plot should be in a cell in the plot section). 
It is possible to play with different configs to get different results.
* Use 'Run on test set with best hyperparameter for each regularization' to
run best params (or any configuration alike) on test set (plots are saved in plots folder)
* Use 'Check training on loaded model' if you like to load a model from
  (if exists) a specific point (we saved models only for validation)
  
Notes:
* The training part was heavily inspired by the Udacity course (PyTorch)