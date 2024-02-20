from pathlib import Path
import logging
import pandas as pd
from datetime import datetime

def document_results(results_dict):
    results_file_path = Path('tests') / 'results.csv'
    logging.info(f"results_file_path: {results_file_path}")
    if results_file_path.exists():
        df = pd.read_csv(results_file_path)
        new_df = pd.DataFrame([results_dict])
        new_df['date'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        df = pd.concat([df, pd.DataFrame([results_dict])])
    else:
        df = pd.DataFrame(results_dict, index=[0])
    df.to_csv(results_file_path, index=False)


def plot_and_save_synthetic(train_n, test_n, test_a, image_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.clf()
    sns.scatterplot(x=test_a[:, 0], y=test_a[:, 1], color='red', label=f'anomaly_test (# {test_a.shape[0]})')
    sns.scatterplot(x=test_n[:, 0], y=test_n[:, 1], color='green', label=f'normal_test (# {test_n.shape[0]})')
    sns.scatterplot(x=train_n[:, 0], y=train_n[:, 1], color='blue', label=f'normal_train (# {train_n.shape[0]})')
    save_path = Path('tests') / 'images'
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path/(image_name + ".png"))

