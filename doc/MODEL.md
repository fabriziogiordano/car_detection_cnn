/parking_lot_classifier
│
├── /data                      # Directory containing all image data
│   ├── /train                 # Training data
│   │   ├── /car               # Training images with cars
│   │   └── /no_car            # Training images without cars
│   ├── /test                  # Testing data
│   │   ├── /car               # Testing images with cars
│   │   └── /no_car            # Testing images without cars
│   └── /preprocessed          # Optional: Preprocessed images (if you choose to save them)
│       ├── /train             # Preprocessed training data
│       │   ├── /car           # Preprocessed training images with cars
│       │   └── /no_car        # Preprocessed training images without cars
│       ├── /test              # Preprocessed testing data
│       │   ├── /car           # Preprocessed testing images with cars
│       │   └── /no_car        # Preprocessed testing images without cars
│
├── model.py                   # Python script defining the CNN model
├── train.py                   # Python script for training the model
├── evaluate.py                # Python script for evaluating the model
├── preprocess.py              # Python script for preprocessing and optionally saving images
├── inference.py               # Python script for running inference on Raspberry Pi
├── car_detection_cnn.pth      # Saved model after training
└── README.md                  # Instructions and documentation