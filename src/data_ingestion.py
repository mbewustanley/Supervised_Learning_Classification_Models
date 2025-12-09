import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


## Dataset source:
# Bock, R. (2004). MAGIC Gamma Telescope [Dataset]. UCI Machine Learning 
# Repository. https://doi.org/10.24432/C52C8B.

cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
         'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']


def load_data(data_url: str, cols : list) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url, names=cols)
        return df
    
    #error handline
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the CSV file from {data_url}.")
        print(e)
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the data.")
        print(e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['class'] = (df['class'] == 'g').astype(int)
        for label in cols[:-1]:
            plt.hist(df[df['class']==1][label], color='blue', label='gamma', alpha=0.7, density=True)
            plt.hist(df[df['class']==0][label], color='red', label='hadron', alpha=0.7, density=True)
            plt.title(label)
            plt.ylabel('probability')
            plt.xlabel(label)
            plt.legend()

            os.makedirs("data/plots", exist_ok=True)
            output_path = os.path.join("data/plots", f"{label}.png")
            plt.savefig(output_path)      # <-- Correct way to save
            plt.clf()  
        return df
    
    #error handling
    except KeyError as e:
        print(f"Error: Missing column {e} in the dataframe.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing.")
        print(e)
        raise

def save_data(df: pd.DataFrame, output_file: str) -> None:
    try:
        os.makedirs("data/raw", exist_ok=True)
        output_path = os.path.join("data/raw", output_file)

        df.to_csv(output_path, index=False)   # <-- save directly to path
        print(f"Data saved to {output_path}")

    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise


def main(data_url, cols):
    try:
        df = load_data(data_url, cols)
        final_df = preprocess_data(df)
        save_data(final_df, "dataset.csv")

    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")


if __name__ == '__main__':
    main('data/input_data/magic04.data', cols)