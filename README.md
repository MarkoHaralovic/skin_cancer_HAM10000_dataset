# HAM10000 Skin Lesion Classifier

7-class skin lesion classifier trained on the [HAM10000 dataset](https://doi.org/10.7910/DVN/DBW86T).

## Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run training

```bash
python train_model.py --config configs/resnet50_ce_ifw.json
```

Any config key can be overridden directly on the command line:

```bash
python train_model.py --config configs/resnet50_ce_ifw.json --epochs 10 --device cpu
```

## Config — paths to change

Open a config file (e.g. `configs/resnet50_ce_ifw.json`) and update these two fields:

```json
"data_path": "C:/path/to/dataverse_files",
"metadata_csv": "HAM10000_metadata",
```

- **`data_path`** — absolute path to the folder that contains `HAM10000_metadata`, `HAM10000_images_part_1/`, `HAM10000_images_part_2/`, etc.
- **`metadata_csv`** — filename of the metadata file inside `data_path`. The HAM10000 download has no `.csv` extension (`HAM10000_metadata`), so leave it as-is unless your file differs.
- **`image_dirs`** — leave as `null`; image folders are auto-detected from `data_path`.

## Dataset layout expected

```
dataverse_files/
├── HAM10000_metadata               
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
├── ISIC2018_Task3_Test_GroundTruth.csv
└── ISIC2018_Task3_Test_Images/
```