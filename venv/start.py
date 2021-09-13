
import zipfile
with zipfile.ZipFile('dataset_n.zip', 'r') as zip_ref:
    zip_ref.extractall()