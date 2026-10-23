# MLflow - Manage ML Experiments

Scripts for trying out MLflow experiment tracking, MLflow Projects, MLflow Models and the Model Registry.
Tested with Python 3.12 and the versions pinned in `requirements.txt` (MLflow 3.17.2).

| Script | Experiment | What it does |
|---|---|---|
| `demo.py` | `Demo_Experiment` | Dummy run: logs two params, a metric, a tag and an artifact (`dummy/example.txt`) |
| `basic_ml.py` | `ML-Model-1` | ElasticNet on the UCI wine-quality data (downloaded at runtime); logs params, metrics and the model |
| `help.py` | `Loan_Prediction_Demo` | Minimal tracking API example: one named run with a tag, params and a metric |
| `loan_prediction.py` | `Loan_prediction` | Grid search for DecisionTree / LogisticRegression / RandomForest on `train.csv`; one run per model with params, metrics, ROC plot and model |
| `test-model.py` | – | Loads a logged model by run ID and predicts |
| `model-serve.py` | – | Loads a registered model by alias and predicts |

Follow the steps in order. Steps 1–2 are all you need for the dummy demo experiments; steps 3–5 cover MLflow Projects, model serving and the Model Registry. Run every command from this folder.

## 1. Set up the environment

With conda:

```bash
conda create -n mlflow-venv python=3.12
conda activate mlflow-venv
pip install -r requirements.txt
```

Or with venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the dummy demo experiments

```bash
python demo.py                      # defaults: --param1 5 --param2 10
python demo.py -p1 3 -p2 4          # a second run to compare against
python basic_ml.py -a 0.5 -l1 0.7   # defaults: --alpha 0.2 --l1_ratio 0.3
python help.py
python loan_prediction.py           # grid search, can take a few minutes on a laptop; prints each model's run ID
```

Then open the UI and compare the runs:

```bash
mlflow ui --port 5000               # open http://127.0.0.1:5000
```

Without a tracking URI, MLflow stores runs in `./mlflow.db` (SQLite) and artifacts in `./mlruns/`, so start `mlflow ui` from this folder. Delete both to start from scratch (they are git-ignored).

If `./mlruns` holds runs from an older MLflow version (folders with `meta.yaml` files), MLflow 3 stops with a "filesystem tracking backend ... is in maintenance mode" error. Move that folder aside, or convert it with `mlflow migrate-filestore`.

## 3. Run as an MLflow Project

`MLProject` runs `loan_prediction.py`. `--experiment-name` must be exactly `Loan_prediction`, the experiment the script logs to; with no name or any other name the run fails with an experiment-mismatch error.

```bash
# builds a conda environment from conda.yaml (conda must be installed)
mlflow run . --experiment-name Loan_prediction

# without conda: build the environment with uv (needs `uv` on PATH, e.g. `pip install uv`) ...
mlflow run . --experiment-name Loan_prediction --env-manager uv

# ... or reuse the active environment from step 1
mlflow run . --experiment-name Loan_prediction --env-manager local

# straight from the git repository
mlflow run https://github.com/cybervava/MLFlow-Manage-ML-Experiments --experiment-name Loan_prediction --env-manager local
```

## 4. Load and serve a logged model

`loan_prediction.py` logs each model under its class name (`DecisionTreeClassifier`, `LogisticRegression`, `RandomForestClassifier`) in its own run and prints the run IDs.

Load a logged model in Python and predict:

```bash
python test-model.py <RUN_ID>       # add -m LogisticRegression or -m DecisionTreeClassifier for those runs
```

Serve it with the local REST server:

```bash
mlflow models serve -m runs:/<RUN_ID>/RandomForestClassifier --port 9000 --env-manager local
```

`--env-manager local` serves from the active environment. Without it, MLflow builds a separate virtualenv, which needs `pyenv`; `--env-manager uv` is a faster alternative.

Generate predictions from another terminal (the same JSON body works in Postman):

```bash
curl --location 'http://127.0.0.1:9000/invocations' \
--header 'Content-Type: application/json' \
--data '{
    "dataframe_split": {
        "columns": [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
            "TotalIncome"
        ],
        "data": [
            [
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
            ]
        ]
    }
}'
```

Response: `{"predictions": [1]}`

MLflow 3 stores logged models apart from the run's artifacts, under `mlruns/<experiment_id>/models/<model_id>/artifacts/`, and saves scikit-learn models in the `skops` format:
```yaml
m-<model_id>/artifacts/
    - MLmodel
    - conda.yaml
    - model.skops
    - python_env.yaml
    - requirements.txt
```

## 5. Tracking server and Model Registry (MySQL)

Install the pure-Python MySQL driver, create an empty database (`CREATE DATABASE db_mlflow;`), then start the server on port 5001. Passing the connection string through `MLFLOW_BACKEND_STORE_URI` keeps the password out of the process list.

```bash
pip install pymysql
export MLFLOW_BACKEND_STORE_URI='mysql+pymysql://<user>:<password>@localhost/db_mlflow'
mlflow server --port 5001 --default-artifact-root $PWD/mlruns
```

The server has no authentication and listens on 127.0.0.1 by default; keep it that way for local testing. `mysqlclient` with `mysql://` URIs also works, but on Linux it needs the MySQL client headers to build. To try the server without MySQL, run `mlflow server --port 5001` on its own: it stores runs in `./mlflow.db` and artifacts in `./mlartifacts/`.

In a second terminal, point the scripts at the server and log the models:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
python loan_prediction.py
```

Register the RandomForestClassifier model as `Prediction_Model_RF` and give the new version the alias `champion` (aliases replace the deprecated `Staging`/`Production` stages). You can do this in the UI, or with:

```bash
python -c "import mlflow; v = mlflow.register_model('runs:/<RUN_ID>/RandomForestClassifier', 'Prediction_Model_RF'); mlflow.MlflowClient().set_registered_model_alias('Prediction_Model_RF', 'champion', v.version)"
```

MLflow 3 logs a warning that the run has no artifacts at that path and registers the logged model instead; that is expected.

Serve the registered model over REST, or load it in-process:

```bash
mlflow models serve -m "models:/Prediction_Model_RF@champion" -p 1234 --env-manager local
python model-serve.py
```

## Notes on MLflow Tracking
`mlflow.set_tracking_uri()` connects to a tracking URI. You can also set the MLFLOW_TRACKING_URI environment variable to have MLflow find a URI from there. In both cases, the URI can either be a HTTP/HTTPS URI for a remote server, a database connection string, or a local path to log data to a directory. With the MLflow version pinned here, the URI defaults to `sqlite:///mlflow.db` (older MLflow versions used the local `mlruns` folder).

`mlflow.get_tracking_uri()` returns the current tracking URI.

`mlflow.create_experiment()` creates a new experiment and returns its ID. Runs can be launched under the experiment by passing the experiment ID to mlflow.start_run. It raises an error if the experiment already exists.

`mlflow.set_experiment()` sets an experiment as active. If the experiment does not exist, creates a new experiment. If you do not specify an experiment in mlflow.start_run(), new runs are launched under this experiment.

`mlflow.start_run()` starts a new run and returns a mlflow.ActiveRun object usable as a context manager for the current run. It raises an error if a run is already active, unless you pass `nested=True`; `mlflow.active_run()` returns the active run. You do not need to call start_run explicitly: calling one of the logging functions with no active run automatically starts a new one.

`mlflow.end_run()` ends the currently active run, if any, taking an optional run status.

`mlflow.log_param()` logs a single key-value param in the currently active run. The key and value are both strings. Use mlflow.log_params() to log multiple params at once.

`mlflow.log_metric()` logs a single key-value metric. The value must always be a number. MLflow remembers the history of values for each metric. Use mlflow.log_metrics() to log multiple metrics at once.

`mlflow.set_tag()` sets a single key-value tag in the currently active run. The key and value are both strings. Use mlflow.set_tags() to set multiple tags at once.

`mlflow.log_artifact()` logs a local file or directory as an artifact, optionally taking an artifact_path to place it in within the run’s artifact URI. Run artifacts can be organized into directories, so you can place the artifact in a directory this way.

`mlflow.log_artifacts()` logs all the files in a given directory as artifacts, again taking an optional artifact_path.

## If you get Port Already in use error while using mlflow
- Get list of Services & PID running
`lsof -i tcp:5000`
- Kill them
`kill -15 <PID>`
