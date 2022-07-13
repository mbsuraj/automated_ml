"""Provide a simple data catalog to act as
the single point of truth for the location of
data.

This catalog assumes the data lake is filesystem based.
In realistic situations, it does not have to be.

TODO: improve by making an abstraction of the
 type of the data (database, file: csv/parquet/…, …)
"""

from src.config import DATA_LAKE


def _resource(zone, key):
    return str(DATA_LAKE / zone / key)


catalog = {
    "clean/test": _resource("clean", "test.csv"),
    "clean/train": _resource("clean", "train.csv"),
    "clean/y": _resource("clean", "y_train.csv"),
    "landing/sample_submission": _resource("landing", "sample_submission.csv"),
    "landing/data_description": _resource("landing", "data_description.txt"),
    "landing/test": _resource("landing", "test.csv"),
    "landing/train": _resource("landing", "train.csv"),
    "business/ridge_predictions": _resource("business", "ridge_predictions.csv"),
    "business/lasso_predictions": _resource("business", "lasso_predictions.csv"),
    "business/elasti_predictions": _resource("business", "elasti_predictions.csv"),
    "business/lr_predictions": _resource("business", "lr_predictions.csv"),
    "business/enet_predictions": _resource("business", "enet_predictions.csv"),
    "business/dlearn_predictions": _resource("business", "dlearn_predictions.csv")

}
