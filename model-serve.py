import mlflow
model_name = 'Prediction_Model_RF'

alias = "champion" # alias set on a registered model version (registry stages are deprecated)
# mlflow.set_tracking_uri("http://127.0.0.1:5001")  # or: export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")

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
