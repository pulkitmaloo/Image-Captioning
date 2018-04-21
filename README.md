# CV-Project

## Flask App
To run the flask app that provides a GUI interface to test the model run

   run ./run_flask.sh

## Generate Bottleneck Features

[bottleneck features](Generate_Bottleneck_Features.ipynb)

Link to bottleneck features: 

- Dataset divided into training, test, and validation sets [link](https://drive.google.com/open?id=1blr5_-9c4x6G5QNgkhLNfNCUVegYQASq)
- [previous link](https://drive.google.com/drive/folders/19FEnwYL8ESA1O1DctG9er7tM7Pe6giuM?usp=sharing)

## Generate one-hot vectors with captions

Create a directory `preprocessed_captions`. The result will be saved here.

[One-hot encode captions](Generate_onehot_encoded_captions.ipynb)

Run the above notebook file to get the preprocessed captions.

## Trained Models

- Train models using this [notebook](Experiment1_save_model.ipynb)
- [link to the saved models](https://drive.google.com/drive/folders/1yxzsLg5OtS-wgR8fY-Y3KUhMCtZFEvvC?usp=sharing)

### Model Descriptions

- `weights.best.VGG16.noDropout.hdf5` : `loss: 3.9174e-08 - acc: 0.6714 - val_loss: 3.9518e-08 - val_acc: 0.6685`
    - emb_size = 150
    - lstm_size = 300
    - batch_size=32
    - epochs=3 / 50

