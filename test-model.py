import argparse
import mlflow

args = argparse.ArgumentParser()
args.add_argument("run_id", help="ID of a run logged by loan_prediction.py")
args.add_argument("--model-name","-m", default="RandomForestClassifier",
                  help="DecisionTreeClassifier, LogisticRegression or RandomForestClassifier")
parsed_args = args.parse_args()
logged_model = f'runs:/{parsed_args.run_id}/{parsed_args.model_name}'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
columns = ["Gender", "Married", "Dependents", "Education", "Self_Employed",
           "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area", "TotalIncome"]
data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]
print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data, columns=columns))}")
