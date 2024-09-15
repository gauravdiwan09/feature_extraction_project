import pandas as pd


def sanity_check(output_file):
    try:
        df = pd.read_csv(output_file)
        assert 'index' in df.columns
        assert 'prediction' in df.columns
        print(f"Sanity check passed for file: {output_file}")
    except Exception as e:
        print(f"Sanity check failed: {e}")

