import numpy as np
from argparse import ArgumentParser
from keras.models import Model, Sequential
from keras.layers import Input, Dense, BatchNormalization, RepeatVector, Concatenate, Merge, Masking
from keras.layers import LSTM, GRU, Embedding, TimeDistributed, Bidirectional
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
# from keras.utils import plot_model

from caption_utils import *


# Since there are 5 captions per image, duplicate the bottleneck features
def duplicate_bottleneck_features(features):
    num_captions = 5 # 5 stands for number of captions per image
    num_rows = features.shape[0] * num_captions 

    features_dup = np.zeros((num_rows, features.shape[1]))
    for i, image in enumerate(features):
        for j in range(num_captions):
            features_dup[i*num_captions + j] = image
    return features_dup 


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def generate_seq(img_input, input_shape):
    if img_input.shape != (1, input_shape):
        img_input = img_input.reshape(1, input_shape)

    assert(img_input.shape == (1, input_shape))
    stop_condition = False
    decoded_sentence = []
    target_seq = np.array([token2idx['<bos>']]).reshape(1, 1)
    states_value = encoder_model.predict(img_input)

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        sampled_char = idx2token[sampled_token_index]
        decoded_sentence += [sampled_char]
        if (sampled_char == '<eos>' or len(decoded_sentence) > 30):
            stop_condition = True
        target_seq = np.array([sampled_token_index]).reshape(1, 1)
        states_value = [h, c]

    return ' '.join(decoded_sentence)


if __name__ == "__main__":
    parser = ArgumentParser(description="Image Captioning")
    parser.add_argument('-m', '--model', type=str, default="VGG16", help="Pretrained model for images")
    parser.add_argument('-es', '--emb_size', type=int, default=300, help="Size of the Word Embedding")
    parser.add_argument('-ls', '--lstm_size', type=int, default=300, help="Size of the lstm")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate used in training")
    parser.add_argument('-dr', '--dropout_rate', type=float, default=0.2, help="Dropout rate in the model")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('-ne', '--n_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('-tm', '--trained_model', type=str, default="test.h5", help="filename to save the trained model")
    parser.add_argument('-em', '--encoder_model', type=str, default="encoder_model.h5", help="filename to save the encoder model")
    parser.add_argument('-dm', '--decoder_model', type=str, default="decoder_model.h5", help="filename to save the decoder model")

    args = parser.parse_args()
    model = args.model
    emb_size = args.emb_size
    lstm_size = args.lstm_size
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    # Load Dataset
    print("\nLoading train_dev_test.npz...")
    all_data = np.load('train_dev_test.npz')

    train_decoder_input = all_data['train_decoder_input']
    train_decoder_target = all_data['train_decoder_target'][:,1:,:]
    validation_decoder_input = all_data['validation_decoder_input']
    validation_decoder_target = all_data['validation_decoder_target'][:,1:,:]
    test_decoder_input = all_data['test_decoder_input']
    test_decoder_target = all_data['test_decoder_target'][:,1:,:]

    if args.model == "VGG16":
        train_encoder_output = all_data['train_encoder_output']
        validation_encoder_output = all_data['validation_encoder_output']
        test_encoder_output = all_data['test_encoder_output']

        input_shape = 512
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.final.hdf5', 
                                               verbose=1, save_best_only=True)
        path_best_weights = 'saved_models/weights.best.VGG16.final.hdf5'
        path_model = "saved_models/" + args.trained_model
        path_encoder_model = "saved_models/" + args.encoder_model
        path_decoder_model = "saved_models/" + args.decoder_model

    elif args.model == "VGG19":
        bottleneck_features = np.load('bottleneck_features/Flicker8k_bottleneck_features_VGG19_avgpooling.npz')
        bottleneck_features_train = bottleneck_features["train"]
        bottleneck_features_validation = bottleneck_features["validation"]
        bottleneck_features_test = bottleneck_features["test"]

        train_encoder_output = duplicate_bottleneck_features(bottleneck_features_train)
        validation_encoder_output = duplicate_bottleneck_features(bottleneck_features_validation)
        test_encoder_output = duplicate_bottleneck_features(bottleneck_features_test)

        del bottleneck_features
        del bottleneck_features_train
        del bottleneck_features_validation
        del bottleneck_features_test

        input_shape = 512
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.final.hdf5', 
                                               verbose=1, save_best_only=True)
        path_best_weights = 'saved_models/weights.best.VGG19.final.hdf5'
        path_model = "saved_models/" + args.trained_model
        path_encoder_model = "saved_models/" + args.encoder_model
        path_decoder_model = "saved_models/" + args.decoder_model

    elif args.model == "ResNet50":
        bottleneck_features = np.load('bottleneck_features/Flicker8k_bottleneck_features_ResNet50_avgpooling.npz')
        bottleneck_features_train = bottleneck_features["train"]
        bottleneck_features_validation = bottleneck_features["validation"]
        bottleneck_features_test = bottleneck_features["test"]

        train_encoder_output = duplicate_bottleneck_features(bottleneck_features_train)
        validation_encoder_output = duplicate_bottleneck_features(bottleneck_features_validation)
        test_encoder_output = duplicate_bottleneck_features(bottleneck_features_test)

        del bottleneck_features
        del bottleneck_features_train
        del bottleneck_features_validation
        del bottleneck_features_test

        input_shape = 2048
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.ResNet50.final.hdf5', 
                                               verbose=1, save_best_only=True)
        path_best_weights = 'saved_models/weights.best.ResNet50.final.hdf5'
        path_model = "saved_models/" + args.trained_model
        path_encoder_model = "saved_models/" + args.encoder_model
        path_decoder_model = "saved_models/" + args.decoder_model

    else:
        exit()

    del all_data

    print("\nTrain Decoder Input", train_decoder_input.shape, train_decoder_input.dtype)
    print("Train Decoder Target", train_decoder_target.shape, train_decoder_target.dtype)
    print("Train Encoder Output", train_encoder_output.shape, train_encoder_output.dtype)

    train_fns_list, dev_fns_list, test_fns_list = load_split_lists()

    train_captions_raw, dev_captions_raw, test_captions_raw = get_caption_split()
    vocab = create_vocab(train_captions_raw)
    token2idx, idx2token = vocab_to_index(vocab)     
    captions_data = (train_captions_raw.copy(), dev_captions_raw.copy(), test_captions_raw.copy())
    train_captions, dev_captions, test_captions = process_captions(captions_data, token2idx)
    
    # Build the model
    vocab_size = len(vocab)
    max_length = train_decoder_target.shape[1]
        
    K.clear_session()

    # Image -> Image embedding
    image_input = Input(shape=(train_encoder_output.shape[1], ), name='image_input')
    print("\nImage input shape: {}".format(image_input.shape))
    img_emb = Dense(emb_size, activation='relu', name='img_embedding')(image_input)
    print("Image embedding shape: {}".format(img_emb.shape))

    # Sentence to Word embedding
    caption_inputs = Input(shape=(None, ), name='caption_input')
    print("\nCaption Input Shape", caption_inputs.shape)
    emb_layer = Embedding(input_dim=vocab_size, output_dim=emb_size, name='Embedding')
    word_emb = emb_layer(caption_inputs)
    print("Word embedding shape: {}".format(word_emb.shape))

    decoder_cell = LSTM(lstm_size, return_sequences=True, return_state=True, name='decoder', dropout=dropout_rate, recurrent_dropout=dropout_rate)
    encoder_states = [img_emb, img_emb]
    decoder_out, _, _ = decoder_cell(word_emb, initial_state = encoder_states)

    output_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))
    decoder_out = output_layer(decoder_out)

    rmsprop = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=1e-6)
    model = Model(inputs=[image_input,caption_inputs], outputs=[decoder_out])

    model.compile(optimizer=rmsprop,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    print("\nTraining model summary:")
    print(model.summary())

    # Start Training the model
    model.fit([train_encoder_output, train_decoder_input], [train_decoder_target],
               validation_data=([validation_encoder_output, validation_decoder_input], [validation_decoder_target]),
               epochs=n_epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=2)

    # Inference
    encoder_model = Model(image_input, encoder_states)
    print("\nInference model - encoder summary:")
    print(encoder_model.summary())

    decoder_state_input_h = Input(shape=(lstm_size, ))
    decoder_state_input_c = Input(shape=(lstm_size, ))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_cell(word_emb, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = output_layer(decoder_outputs)
    decoder_model = Model([caption_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    print("\nInference model - decoder summary:")
    print(decoder_model.summary())

    # Load the model with the best validation loss
    model.load_weights(path_best_weights)

    print("\nShowing results for 20 test images:")
    print("<Generated Sequence>")
    print('*'*30)
    print("Five ground truth captions, <bos> stands for beginning of sentence and <eos> stands for end of sentence\n")

    for i in range(20):
        print(generate_seq(test_encoder_output[i*5, :], input_shape))
        print('*'*30)
        for j in range(5):
            print(intseq_to_caption(idx2token, test_captions[test_fns_list[i]][j]))
        print()

    # Save the models
    model.save(path_model)
    encoder_model.save(path_encoder_model)
    decoder_model.save(path_decoder_model)
