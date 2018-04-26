# Image Captioning using Deep Learning

## Flask App
You can test our model in your own computer using the flask app.
To run the flask app that provides a GUI interface, simply clone our repository and run flask.

```./run_flask.sh```

## If you want to reproduce everything (preprocessing, training, ...) then, follow the steps below:

## Step 1: Get data

1. Make an empty directory `data`.
2. Download flickr8K data. Then save the folders in `data`.

## Step 2: Image processing - Generate bottleneck features

1. Make an empty directory `bottleneck_features`.
2. Run the `preprocessing1_bottleneck_features.py`. It will generate bottleneck features for pre-trained models VGG16, VGG19, and ResNet50 and then save them in `bottleneck_features` directory.
3. (Optional) It may take a while to generate the bottleneck features. You can get those files in this [link](https://drive.google.com/open?id=1blr5_-9c4x6G5QNgkhLNfNCUVegYQASq).

## Step 3: Generate one-hot vectors with captions

Create a directory `preprocessed_captions`. The result will be saved here.

[One-hot encode captions](Generate_onehot_encoded_captions.ipynb)

Run the above notebook file to get the preprocessed captions.

## Trained Models

- Train models using this [notebook](Experiment1_save_model.ipynb)
- [link to the saved models](https://drive.google.com/drive/folders/1yxzsLg5OtS-wgR8fY-Y3KUhMCtZFEvvC?usp=sharing)

### Model Descriptions

- `weights.best.VGG16.noDropout.hdf5` : `loss: 3.9174e-08 - acc: 0.6714 - val_loss: 3.9518e-08 - val_acc: 0.6685`
    - emb_size = 300
    - lstm_size = 300
    - batch_size = 32
    - epochs = 3 / 50

