*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

*TODO:* Write a short introduction to your project.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset
### Context

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.

### Overview
The data is provided via the following Kaggle source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

The data is provided as a .csv file and ist structured as followed.

Attribute Information:
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

### Task
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

### Access
The data is provided via the following Kaggle source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

## Automated ML
Configuration and settings used for the Automated ML experiment are described in the table below:
|        Configuration       |                                                               Description                                                               |      Value     |
|:--------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|:--------------:|
| experiment_timeout_minutes | This is used as an exit criteria, it defines how long, in minutes, your experiment should continue to run                               | 20             |
| max_concurrent_iterations  | Represents the maximum number of iterations that would be executed in parallel                                                          | 5              |
| primary_metric             | The metric that Automated Machine Learning will optimize for model selection                                                            | accuracy       |
| task                       | The type of task to run. Values can be 'classification',  'regression', or 'forecasting' depending on the type of automated ML  problem | classification |
| compute_target             | The compute target to run the experiment on                                                                                             | trainCluster   |
| training_data              | Training data, contains both features and label columns                                                                                 | ds             |
| label_column_name          | The name of the label column                                                                                                            | Class          |
| n_cross_validations        | No. of cross validations to perform                                                                                                     | 5              |

### Results
In our experiment we found out SparseNormalizer GradientBoosting to be the best model based on the accuracy metric. The accuracy score for this models was 0.9882352941176471.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

### AutoML Screenshots


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

### AutoML Screenshots


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

