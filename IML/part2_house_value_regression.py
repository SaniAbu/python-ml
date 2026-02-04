import pickle
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

from part1_nn_lib import MultiLayerNetwork, Trainer


class Regressor:

    def __init__(self, x, nb_epoch = 1000, learning_rate = 0.004, batch_size = 256, l1 = 256, l2 = 128, l3 = 64):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.column_means_input = None
        self.column_stds_input = None
        self.column_means_input_np = None
        self.column_stds_input_np = None
        self.column_means_output = None
        self.column_stds_output = None

        self.label_binarizer = None

        self.network = MultiLayerNetwork(self.input_size, [self.l1, self.l2, self.l3, 1], ["relu", "relu", "relu", "identity"])
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def get_params(self, deep=True):
        return {
            "nb_epoch": self.nb_epoch,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "l1": self.l1,
            "l2": self.l2,
            "l3": self.l3,
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        modified_x = x.drop("ocean_proximity", axis=1)

        if training:
            self.column_means_input = modified_x.mean()
            self.column_stds_input = modified_x.std().replace(0, 1)
            self.column_means_input_np = self.column_means_input.to_numpy()
            self.column_stds_input_np = self.column_stds_input.to_numpy()
            self.column_means_output = y.mean() if y is not None else None
            self.column_stds_output = y.std().replace(0, 1) if y is not None else None

            self.label_binarizer = LabelBinarizer()
            self.label_binarizer.fit(x["ocean_proximity"].unique())

        modified_x = modified_x.fillna(self.column_means_input)
        y = y.fillna(self.column_means_output) if y is not None else None
        
        modified_x = modified_x.sub(self.column_means_input).div(self.column_stds_input)
        y = y.sub(self.column_means_output).div(self.column_stds_output) if y is not None else None

        binary_labels = self.label_binarizer.transform(x["ocean_proximity"])
        modified_x = modified_x.join(pd.DataFrame(binary_labels, columns=self.label_binarizer.classes_, index = modified_x.index)) 

        return modified_x.to_numpy(), (y.to_numpy() if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        trainer = Trainer(self.network, self.batch_size, self.nb_epoch, self.learning_rate, "mse", True)
        self.trainer = trainer
        trainer.train(X, Y)

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        result = self.network(X)
        result = result * self.column_stds_output.to_numpy() + self.column_means_output.to_numpy()
        return result

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        y_pred = self.network(X)

        Y = Y * self.column_stds_output.to_numpy() + self.column_means_output.to_numpy()
        y_pred = y_pred * self.column_stds_output.to_numpy() + self.column_means_output.to_numpy()
        return np.sqrt(mean_squared_error(Y, y_pred))


    def plot_training_history(self, savepath=None):
        """Plot training (and validation if available) loss per epoch.

        The Trainer stores history in `self.trainer.train_loss_history` and
        `self.trainer.val_loss_history` (if validation was provided to
        `Trainer.train`). Call this after `fit()`.
        """
        if not hasattr(self, "trainer"):
            raise RuntimeError("No trainer found. Call fit() before plotting history.")

        train_loss = getattr(self.trainer, "train_loss_history", None)
        val_loss = getattr(self.trainer, "val_loss_history", None)

        if train_loss is None:
            raise RuntimeError("Trainer does not contain train_loss_history")

        epochs = list(range(1, len(train_loss) + 1))

        plt.figure()
        plt.plot(epochs, train_loss, label="train loss")
        if val_loss and len(val_loss) == len(train_loss):
            plt.plot(epochs, val_loss, label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training loss per epoch")
        plt.legend()
        plt.grid(True)
        # default save path when interactive display is not available
        if savepath is None:
            savepath = "training_history.png"
        plt.savefig(savepath)
        plt.close()
        print(f"Saved training history to {savepath}")


    def plot_predictions(self, x, y=None, savepath=None):
        """Plot predicted vs actual scatter and residual histogram.

        Args:
            x {pd.DataFrame} -- input features
            y {pd.DataFrame or None} -- true targets; if None, only predictions are shown
        """
        y_pred = self.predict(x)

        # ensure arrays
        y_pred = np.asarray(y_pred).reshape(-1)
        if y is None:
            plt.figure()
            plt.plot(y_pred, label="predictions")
            plt.title("Predictions")
            plt.ylabel("Target value")
            plt.legend()
            if savepath is None:
                savepath = "predictions.png"
            plt.savefig(savepath)
            plt.close()
            print(f"Saved predictions plot to {savepath}")
            return

        # extract y values
        y_true = y.values.reshape(-1) if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else np.asarray(y).reshape(-1)

        # scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, "r--", label="y = y_pred")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        plt.title(f"Predicted vs True (RMSE={rmse:.3f})")
        plt.legend()
        plt.grid(True)
        if savepath is None:
            savepath = "predictions_vs_true.png"
        plt.savefig(savepath)
        plt.close()
        print(f"Saved predicted vs true plot to {savepath}")

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

class SciKitRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, nb_epoch=1000, learning_rate=0.004, batch_size=256, l1=256, l2=128, l3=64):
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.regressor = None
    
    def fit(self, X, y):
        self.regressor = Regressor(X, nb_epoch=self.nb_epoch, learning_rate=self.learning_rate, batch_size=self.batch_size, l1=self.l1, l2=self.l2, l3=self.l3)

        self.regressor.fit(X, y)
        return self
    
    def predict(self, X):
        return self.regressor.predict(X)



def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def perform_hyperparameter_search(x, y): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    param_distributions = {
        'nb_epoch': [i for i in range(30, 100, 10)],
        'learning_rate': [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016],
        'batch_size': [16, 32, 64, 128, 256],
        'l1': [i for i in range(64, 512, 2)],
        'l2': [i for i in range(64, 512, 2)],
        'l3': [i for i in range(64, 512, 2)],
    }

    regressor = SciKitRegressor()
    # random_search = HalvingRandomSearchCV(estimator=regressor, param_distributions=param_distributions, cv=4, verbose=2, n_jobs=-1, scoring="neg_root_mean_squared_error")
    random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_distributions, n_iter=20, cv=4, verbose=2, n_jobs=-1, scoring="neg_root_mean_squared_error")
    random_search.fit(x, y)

    print("Best Hyperparameters: ", random_search.best_params_)

    best_regressor = random_search.best_estimator_
    print("Best RMSE (CV):", -random_search.best_score_)
    rmse_test = best_regressor.regressor.score(x, y)
    print("Train RMSE:", rmse_test)

    return random_search.best_params_

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    X = data.loc[:, data.columns != output_label]
    Y = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=None
    )

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    # regressor = Regressor(x_train, nb_epoch = 20)
    # regressor.fit(x_train, y_train)
    # save_regressor(regressor)

    # Error
    # error = regressor.score(x_train, y_train)
    # print(f"\nRegressor error: {error}\n")

    best_params = perform_hyperparameter_search(x_train, y_train)

    regressor = Regressor(x_train)
    regressor.set_params(**best_params)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    regressor.plot_training_history()
    regressor.plot_predictions(x_train, y_train)

    error = regressor.score(x_test, y_test)
    print(f"\nRegressor test error: {error}\n")

if __name__ == "__main__":
    example_main()

