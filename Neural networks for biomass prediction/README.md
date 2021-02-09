# Neural Networks for biomass prediction

This is the folder containing the Matlab codes for data preprocessing and data visualization,  as well as the python codes for developing the neural networks used in the paper:
**Predictable programming of branching patterns based on an optimization principle**, 
Nan Luo, Shangying Wang, Jia Lu, Xiaoyi Ouyang, Lingchong You.

* We used Python version 3.6.5 and implement TensorFlow 1.11.0 for neural network design and trainings/validations/tests. 

* During data preprocessing, we used min-max scaling to normalize all the input parameters to be within the region 0-1. Code for data preprocessing (using Matlab): data_cleaning.m

* The neural networks that we developed have nine input parameters and one output corresponding to the biomass of the cell. 
 
* Between the input layer and output layer, we have four fully connected layers with 512, 512, 256, and 128 nodes, respectively.  

* We used the HE initialization method and leaky RELU as activation function with the negative slope α=0.2. 

* We chose the initial learning rate to be 0.001

* We used the Adam optimization algorithm to adaptive moment estimation and gradient clipping to prevent exploding gradients. 

* We trained 3 neural networks independently (train 3 NNs using code "NN_predict_biomass.py" in the folder /NN1, /NN2 and /NN3. During the training process, the models are saved in the "/save" folder. After NNs are trained, the final models are stored in the folder "/saved_model".

* We implemented the ensemble method to get the finalized prediction (Wang et al., 2019#), i.e., use the 3 NNs to make independent predictions and choose the one that has the lowest rmse (root mean squared error) when compared to the other predictions. This method can further reduce prediction errors and make neural network predictions more accurate and reliable. The code name for this task is: "NN_ensemble_test_3ensembles.py"

* We compared the NN predictions with the ground truth (simulation output) for the training and test data sets. Code name: "correlation_pred_ensemble_3NNs.m"

* Other than use the ensemble method to predict the biomass for the training and test data sets, we also try to screen a broader parameter space for the prediction. "screening_NN_predict_biomass_ensemble.py" and "screening_NN_predict_biomass_ensemble_append.py" are used for this purpose.

* task_training.sh is the shell script for running the python codes in a computer bash cluster. 



#Below is the BibTex of the paper cited for the ensemble method:

@article{wang2019massive,
  title={Massive computational acceleration by using neural networks to emulate mechanism-based biological models},
  author={Wang, Shangying and Fan, Kai and Luo, Nan and Cao, Yangxiaolu and Wu, Feilun and Zhang, Carolyn and Heller, Katherine A and You, Lingchong},
  journal={Nature communications},
  volume={10},
  number={1},
  pages={1--9},
  year={2019},
  publisher={Nature Publishing Group}
}
