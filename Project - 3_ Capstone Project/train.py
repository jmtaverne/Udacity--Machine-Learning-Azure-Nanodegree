from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

data_loc = "https://raw.githubusercontent.com/jmtaverne/Udacity--Machine-Learning-Azure-Nanodegree/main/Project%20-%203_%20Capstone%20Project/healthcare-dataset-stroke-data.csv"
ds = TabularDatasetFactory.from_delimited_files(data_loc)


#Save model for current iteration

run = Run.get_context()
  
x_df = ds.to_pandas_dataframe().dropna()
x_df = x_df.drop(["gender","ever_married","work_type","Residence_type","smoking_status"],axis=1)
x_df = x_df.loc[x_df["bmi"] != "N/A"]
y_df = x_df.pop("stroke")

# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=123)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the forest.")
    parser.add_argument('--max_depth', type=int, default=1, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")

    args = parser.parse_args()

    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("max_depth:", np.int(args.max_depth))

    model = RandomForestClassifier(n_estimators=int(args.n_estimators), max_depth=int(args.max_depth)).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    #Save model for current iteration, also include the value for C and max_iter in filename, random_state=
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperDrive_{}_{}'.format(args.n_estimators,args.max_depth))

if __name__ == '__main__':
    main()