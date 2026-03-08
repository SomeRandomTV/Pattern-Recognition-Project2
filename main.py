import sys
import cvxopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# This file implements the Soft-Margin Support Vector Machine 
# The Soft-Margin SVM is implemented in it's Dual Form
# This SVM will be tested at C = 0.1 and C = 100
#   Where: C is the upper bound of lambdas 0 < 	λ < C
#
# C is used to determine what is prioitiezed more: 
#   Increase C: Focus Reducing Misclassifications thus reducing margin
#   Decrease C: Focus on Increasing Margins thus increasing misclassifications
# 
# After implementations I will compare my implementation to a prebuilt/futher optimized 
# To see how poorly ts scales


DEFAULT_DATA_PATH = Path.cwd() / "Proj2&3DataSet.xlsx"

class SoftMarginSVM:

    def __init__(self, data_path: Path):

        if data_path is None:
            print(f"WARNING PATH {data_path} was not found. Using backup ./Project2&3DataSet.xlsx")
            data_path = DEFAULT_DATA_PATH
        
        self.data = self._load_data(data_path=data_path)
        self.inputs = self.data[["x_1", "x_2"]]
        self.labels = self.data["class"]
        self.num_samples, self.num_features = self.inputs.shape
        self.lambdas = np.zeros(self.num_samples) 



    @staticmethod
    def _load_data(data_path: Path):
        
        try:
            df = pd.read_excel(data_path)
            df.columns = ["x_1", "x_2", "class"]

            return df

        except FileNotFoundError as e:
            raise(f"ERROR: {e}")
            return None
        except Exception as e: 
            raise(f"ERROR: Uncaught Exception: {e}")
            return None
    

    def print_df_info(self):

        print("========== Dataset Details ==========")
        print(f"-- DF Head \n {self.data.head()}\n")
        print(f"-- DF Details\n {self.data.describe()}\n")

    def _kernel(x, z):
        
        pass        

    def _compute_gram_matrix(X):

        pass


    def fit(X,y, type="linear", C=0.1):
        pass
    

    def plot_data(self):
        plt.scatter(x=self.inputs["x_1"], y=self.inputs["x_2"], c=self.labels)
        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.show()




def main():

    if len(sys.argv) != 2:
        print(f"ERROR: Not enough args. Expected 3 but got {len(args)}")
        print(f"USAGE: python main.py <path/to/data.xlsx>")
        exit(1)

    path_to_data = Path(sys.argv[1])
    print(f"-> Working with Data at: {path_to_data}")


    svm = SoftMarginSVM(data_path=path_to_data)
    svm.print_df_info()
    svm.plot_data()

if __name__ == "__main__":
    main()









