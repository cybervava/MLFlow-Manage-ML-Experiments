import mlflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or: export MLFLOW_TRACKING_URI=http://127.0.0.1:5000


# set_experiment() creates the experiment on the first run and reuses it afterwards
# (create_experiment() raises an error if the experiment already exists).
# Not 'Loan_Prediction': MySQL compares names case-insensitively, so that would
# collide with the 'Loan_prediction' experiment used by loan_prediction.py.
mlflow.set_experiment('Loan_Prediction_Demo')


with mlflow.start_run(run_name='DecisionTreeClass') as run:
    mlflow.set_tag("version","1.0.0")

    n_estimators=10
    criterion='gini'
    mlflow.log_param('n_estimators',n_estimators)
    mlflow.log_param('criterion',criterion)
    mlflow.log_metric('accuracy',0.9)
