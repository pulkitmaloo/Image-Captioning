# Image Captioning using Deep Learning

You can find the details for our experiments in the report.

## Flask App
You can test our model in your own computer using the flask app.
To run the flask app that provides a GUI interface, simply clone our repository and run flask.

```./run_flask.sh```

## If you want to reproduce everything (preprocessing, training, ...) then, follow the steps below:

### Step 0: Setting up the environment

To allow you to quickly reproduce our results, we are sharing the `environment.yml` file in our github repository. You can simply create the environment using the `environment.yml` file.

```
conda env create -f environment.yml
```

Conda environment name is `tensorflow-3.5` which is using Python 3.5 . Details regarding creating the environment can be found here: [conda link](https://conda.io/docs/user-guide/tasks/manage-environments.html#create-env-from-file)

### Step 1: Get data

1. Make an empty directory `data`.
2. Download flickr8K data. Then save the folders in `data`.

### Step 2: Image preprocessing - Generate bottleneck features

1. Make an empty directory `bottleneck_features`.
2. Run the `preprocessing1_bottleneck_features.py`. It will generate bottleneck features for pre-trained models VGG16, VGG19, and ResNet50 and then save them in `bottleneck_features` directory.

```
python preprocessing1_bottleneck_features.py
```

3. (Optional) It may take a while to generate the bottleneck features. You can get those files in this [link](https://drive.google.com/open?id=1blr5_-9c4x6G5QNgkhLNfNCUVegYQASq).

### Step 3: Caption preprocessing - Word to vector

1. Make an empty directory `preprocessed_captions`
2. Run the `preprocessing2_word_to_vector.py`. It will generate onehot encoded word vectors and then save them in `preprocessed_captions` directory.

```
python preprocessing2_word_to_vector.py
```

### Step 4: Combine bottleneck features and processed captions

Run preprocessing3_data_for_training_model.py. It will generate numpy arrays to be used in training the model. All of the numpy arrays are saved in `train_dev_test.npz` file.

```
python preprocessing3_data_for_training_model.py
```

### Step 5: Train the model 

Run `main.py` to train and save the model. Here are some of the commands that trains, and saves models. 

```
python main.py -m VGG16 -ne 10 -tm test_VGG16.h5 -em encoder_model_VGG16.h5 -dm decoder_model_VGG16.h5
python main.py -m VGG19 -ne 10 -tm test_VGG19.h5 -em encoder_model_VGG19.h5 -dm decoder_model_VGG19.h5
python main.py -m ResNet50 -ne 10 -tm test_ResNet50.h5 -em encoder_model_ResNet50.h5 -dm decoder_model_ResNet50.h5
```

There are several options for `main.py`.

```
(tensorflow-3.5) ubuntu@ip-172-31-62-39:~/CV-Project$ python main.py -h
Using TensorFlow backend.
usage: main.py [-h] [-m MODEL] [-es EMB_SIZE] [-ls LSTM_SIZE]
               [-lr LEARNING_RATE] [-dr DROPOUT_RATE] [-bs BATCH_SIZE]
               [-ne N_EPOCHS] [-tm TRAINED_MODEL] [-em ENCODER_MODEL]
               [-dm DECODER_MODEL]

Image Captioning

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Pretrained model for images
  -es EMB_SIZE, --emb_size EMB_SIZE
                        Size of the Word Embedding
  -ls LSTM_SIZE, --lstm_size LSTM_SIZE
                        Size of the lstm
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate used in training
  -dr DROPOUT_RATE, --dropout_rate DROPOUT_RATE
                        Dropout rate in the model
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Training batch size
  -ne N_EPOCHS, --n_epochs N_EPOCHS
                        Number of epochs for training
  -tm TRAINED_MODEL, --trained_model TRAINED_MODEL
                        filename to save the trained model
  -em ENCODER_MODEL, --encoder_model ENCODER_MODEL
                        filename to save the encoder model
  -dm DECODER_MODEL, --decoder_model DECODER_MODEL
                        filename to save the decoder model
```


### Step 6: Calculate bleu scores for models `VGG16`, `VGG19`, and `ResNet50`

Calculate bleu scores for test images. In order to calculate bleu scores for three models, you need to train each model first, and save the encoder and decoder models as in `Step 5`. 

```
python calculate_bleu_scores.py
```
