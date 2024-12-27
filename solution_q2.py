import pandas as pd
from sklearn.model_selection import train_test_split


## helper functions
# q2.1) Compute the CPTs using the training data
def compute_CPTs(data):
    # print(data)
    total_data_count = 0
    diabetes_false_count = 0
    diabetes_true_count = 0

    prob_diabetes = {
        True: 0,
        False: 0
    }
    prob_glucose_given_Y = {}
    prob_BP_given_Y = {}
    
    # finidng P(Y)
    for i in data["diabetes"]:
        total_data_count += 1
        if i == 0:
            prob_diabetes[i] += 1
            diabetes_false_count += 1
        else:
            prob_diabetes[i] += 1
            diabetes_true_count += 1

    prob_diabetes[0] = prob_diabetes[0] / total_data_count
    prob_diabetes[1] = prob_diabetes[1] / total_data_count
    

    # finding P(X1 | Y)
    diabetes_count_per_glucoseLV = {}
    for index, value in data["glucose"].items():
        if value not in diabetes_count_per_glucoseLV:
            diabetes_count_per_glucoseLV[value] = {0: 0, 1: 0}
        diabetes_count_per_glucoseLV[value][data["diabetes"][index]] += 1
    
    for x1, y_count in diabetes_count_per_glucoseLV.items():
        prob_glucose_given_Y[(0, x1)] = y_count[0] / diabetes_false_count
        prob_glucose_given_Y[(1, x1)] = y_count[1] / diabetes_true_count
    

    # finding P(X2 | Y)
    diabetes_count_per_BPLV = {}
    for index, value in data["bloodpressure"].items():
        if value not in diabetes_count_per_BPLV:
            diabetes_count_per_BPLV[value] = {0: 0, 1: 0}
        diabetes_count_per_BPLV[value][data["diabetes"][index]] += 1
    
    for x2, y_count in diabetes_count_per_BPLV.items():
        prob_BP_given_Y[(0, x2)] = y_count[0] / diabetes_false_count
        prob_BP_given_Y[(1, x2)] = y_count[1] / diabetes_true_count


    return prob_diabetes, prob_glucose_given_Y, prob_BP_given_Y

# q2.2) Implementing inference by enumeration
def inference(data, p_y, p_x1_y, p_x2_y):

    # q2.2.1) code for finding the inference query P(Y | X1, X2)
    inference_table = {}

    for _, row in data.iterrows():
        x1 = row["glucose"]
        x2 = row["bloodpressure"]

        for y in p_y:
            # P(Y | X1, X2) = P(Y) * P(Y | X1) * P(Y | X2)
            # Note: if testing data has the x1 and x2 values that were not in training data, assign 0 as their probability
            inference_table[(y, x1, x2)] = p_y[y] * p_x1_y.get((y, x1), 0) * p_x2_y.get((y, x2), 0) 

    return inference_table

def predict(data, inference_table):
    
    prediction_table = {}

    for _, row in data.iterrows():
        x1 = row["glucose"]
        x2 = row["bloodpressure"]

        prediction_table[(x1, x2)] = 1 if (inference_table[(1, x1, x2)] > inference_table[0, x1, x2]) else 0

    return prediction_table

def calculate_accuracy(data, prediction_table):

    correct_predictions = 0
    total_predictions = 0
    
    for _, row in data.iterrows():
        y = row["diabetes"]
        x1 = row["glucose"]
        x2 = row["bloodpressure"]

        if (y == prediction_table[(x1, x2)]):
            correct_predictions += 1
        total_predictions += 1

    return correct_predictions / total_predictions


## loading and splitting the dataset
# Load the dataset
data = pd.read_csv("Naive-Bayes-Classification-Data.csv")

# Split the data into training and testing sets
training_data, testing_data = train_test_split(data, test_size=0.3, stratify=data["diabetes"])


## answers for q2
p_y, p_x1_y, p_x2_y = compute_CPTs(training_data)
print(f"CPT for P(Y): {p_y}\n") # q2.1.1, each key refers to either Y is true or false
print(f"CPT for P(X1 | Y): {p_x1_y}\n") # q2.1.2, each key is in a format of (Y, X1)
print(f"CPT for P(X2 | Y): {p_x2_y}\n") # q2.1.3, each key is in a format of (Y, X2)

inference_table = inference(testing_data, p_y, p_x1_y, p_x2_y)
print(f"inference table: {inference_table}\n") # q2.2.2) inference table

prediction_table = predict(testing_data, inference_table)
print(f"prediction table: {prediction_table}\n") # q2.3.1) prediction table

accuracy = calculate_accuracy(testing_data, prediction_table)
print(f"Accuracy: {accuracy * 100}%\n") # q2.3.2) accuracy
