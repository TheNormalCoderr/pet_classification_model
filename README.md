# Pet Classification Model

This project classifies pet breeds from the Oxford-IIIT Pet dataset.

It contains two implementations:

- `basic_model/` - a small custom CNN built from scratch
- `pretrained_model/` - a transfer learning version using ResNet50

## Dataset

The project uses the Oxford-IIIT Pet image dataset stored in:

`classification_pet_faces/oxford-iiit-pet/images`

Images are resized to `224 x 224` and labels are derived from the filename pattern.

## Project Structure

```text
classification_pet_faces/
├── basic_model/
│   ├── main.py
│   └── training_validation_test_results.csv
├── pretrained_model/
│   ├── pretrained_model.py
│   └── pretrained_model_results.csv
├── oxford-iiit-pet/
└── README.md
```

## Requirements

Install the dependencies in your virtual environment:

```bash
pip install tensorflow numpy pandas scikit-learn
```

If you are using the same environment as this project, make sure the `venv` is activated first.

## How to Run

### Basic CNN

```bash
python3 classification_pet_faces/basic_model/main.py
```

### ResNet50 Transfer Learning Model

```bash
python3 classification_pet_faces/pretrained_model/pretrained_model.py
```

## What the scripts do

Both scripts:

1. Load the image filenames from the dataset
2. Convert filenames into breed labels
3. Build `features_array` and `labels_array`
4. Split the data into train, validation, and test sets
5. Train the model
6. Evaluate on the test set
7. Save training, validation, and test results to CSV

## Output Files

- `basic_model/training_validation_test_results.csv`
- `pretrained_model/pretrained_model_results.csv`

These files store:

- epoch-by-epoch training loss and accuracy
- validation loss and accuracy
- final test loss and test accuracy

## Notes

- The basic model is a simple CNN and is easier to understand.
- The pretrained model uses ResNet50 and usually performs much better, but it takes longer to train.
- Both models use `sparse_categorical_crossentropy`, so the labels stay as integer class codes.

## Learning Goal

This project was built as a beginner-friendly image classification workflow to understand:

- dataset preparation
- train/validation/test splitting
- data augmentation
- overfitting
- transfer learning

## License

No license has been added yet.

## BY 

Amiteshwar Singh
