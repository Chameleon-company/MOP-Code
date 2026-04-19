import json
import pandas as pd
import os


def save_as_json(data: list, output_path: str) -> None:
    """
    Save extracted results as JSON.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_as_csv(data: list, output_path: str) -> None:
    """
    Save extracted results as CSV.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding="utf-8")


def ensure_folder(folder_path: str) -> None:
    """
    Create folder if it does not exist.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)