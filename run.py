from src.autoprognosis_m import AutoprognosisM
from src.utils.utils import (
    dict_to_namespace,
)
import pandas as pd
import argparse
import json

from sklearn.metrics import roc_auc_score


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def main_trainer(args):
    pipeline_config = dict_to_namespace(json.load(open(args.config_file, "r")))

    # Update globals.base_experiment_dir and globals.base_checkpoint_dir
    fold_string = args.train_csv.split("/")[-2]
    disease_string = args.train_csv.split("/")[-4]
    pipeline_config.globals.base_experiment_dir = f"{pipeline_config.globals.base_experiment_dir}{disease_string}/{fold_string}/"
    pipeline_config.globals.base_checkpoint_dir = f"{pipeline_config.globals.base_checkpoint_dir}{disease_string}/{fold_string}/"

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    train_df["diagnostic"] = train_df["diagnostic"].astype(str)
    val_df["diagnostic"] = val_df["diagnostic"].astype(str)
    test_df["diagnostic"] = test_df["diagnostic"].astype(str)

    print("")
    print("------------------------------------------------------------------------------------")
    print(f"------------------- Initializing Autoprognosis-M with {len(pipeline_config.configurations)} models -------------------")
    print("------------------------------------------------------------------------------------")
    print("")
    autoprognosisM = AutoprognosisM(pipeline_config)
    print("")
    print("------------------------------------------------------------------------------------")
    print(f"------------------- Running Autoprognosis-M with {len(pipeline_config.configurations)} models -------------------")
    print("------------------------------------------------------------------------------------")
    print("")
    autoprognosisM.run(train_df=train_df, val_df=val_df, force=args.force)
    print("")
    print("------------------------------------------------------------------------------------")
    print(f"------------------- Fit Autoprognosis-M with {len(pipeline_config.configurations)} models -------------------")
    print("------------------------------------------------------------------------------------")
    print("")
    weights = autoprognosisM.fit(df=val_df, target_metric=args.target_metric)
    print("")
    print("------------------------------------------------------------------------------------")
    print(f"------------------- Predicting with Autoprognosis-M with {len(pipeline_config.configurations)} models -------------------")
    print("------------------------------------------------------------------------------------")
    print("")
    ensemble_predictions_df, auc = autoprognosisM.predict(df=test_df, weights=weights, return_probs=True)
    ensemble_predictions_df = ensemble_predictions_df.merge(
        test_df[["diagnostic", "img_id"]], on="img_id", how="left"
    )
    ensemble_predictions_df.to_csv(
        f"{'/'.join(args.test_csv.split('/')[:-1])}/ensemble_predictions.csv",
        index=False,
    )
    # Store auc
    with open(
        f"{'/'.join(args.test_csv.split('/')[:-1])}/ensemble_auc.txt", "w"
    ) as f:
        f.write(f"AUROC: {auc}\n")

    return ensemble_predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoprognosis-M")
    parser.add_argument(
        "-c",
        "--config-file",
        type=str,
        default="PAD-UFES/config.json",
        help="Path to the config file",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/train.csv",
        help="Path to the train csv file",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/val.csv",
        help="Path to the val csv file",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/test.csv",
        help="Path to the test csv file",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force retraining",
    )
    parser.add_argument(
        "-t",
        "--target-metric",
        type=str,
        default="AUROC",
        choices=["Accuracy", "Bal. Acc.", "AUROC", "F1 Score", "Matt. Corr."],
        help="Target metric for model selection",
    )
    args = parser.parse_args()
    predictions_df = main_trainer(args)
    print(predictions_df)
