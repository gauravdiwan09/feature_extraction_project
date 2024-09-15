import requests
from pathlib import Path

def download_images(csv_file, output_dir='dataset/images'):
    import pandas as pd
    df = pd.read_csv(csv_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx, row in df.iterrows():
        url = row['image_link']
        image_id = row['index']
        response = requests.get(url)
        if response.status_code == 200:
            with open(f"{output_dir}/{image_id}.jpg", 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download image {image_id} from {url}")

