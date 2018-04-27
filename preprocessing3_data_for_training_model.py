import numpy as np
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

def captions_onehot_split(captions_onehot):
    """ returns decoder input data and decoder target data """
    return captions_onehot[:, :-1, :], captions_onehot[:, :, :]

print("Loading bottleneck features...")
bottleneck_features = np.load('bottleneck_features/Flicker8k_bottleneck_features_VGG16_avgpooling.npz')
bottleneck_features_train = bottleneck_features["train"]
bottleneck_features_validation = bottleneck_features["validation"]
bottleneck_features_test = bottleneck_features["test"]

print("Duplicating images...")
bottleneck_features_train_dup = duplicate_bottleneck_features(bottleneck_features_train)
bottleneck_features_validation_dup = duplicate_bottleneck_features(bottleneck_features_validation)
bottleneck_features_test_dup = duplicate_bottleneck_features(bottleneck_features_test)

# Word Embedding
train_fns_list, dev_fns_list, test_fns_list = load_split_lists()

train_captions_raw, dev_captions_raw, test_captions_raw = get_caption_split()
vocab = create_vocab(train_captions_raw)
token2idx, idx2token = vocab_to_index(vocab)
captions_data = (train_captions_raw.copy(), dev_captions_raw.copy(), test_captions_raw.copy())
train_captions, dev_captions, test_captions = process_captions(captions_data, token2idx)

# Caption Preprocessing
print("Loading preprocessed captions...")
captions_onehot = np.load('preprocessed_captions/Flicker8k_onehot_'+str(len(vocab))+'_words.npz')
train_captions_onehot = captions_onehot["train"]
validation_captions_onehot = captions_onehot["validation"]
test_captions_onehot = captions_onehot["test"]

train_captions_onehot = train_captions_onehot.astype(np.float32)
validation_captions_onehot = validation_captions_onehot.astype(np.float32)
test_captions_onehot = test_captions_onehot.astype(np.float32)

print("Generating Decorder input and target data")

train_decoder_input, train_decoder_target = captions_onehot_split(train_captions_onehot)
validation_decoder_input, validation_decoder_target = captions_onehot_split(validation_captions_onehot)
test_decoder_input, test_decoder_target = captions_onehot_split(test_captions_onehot)

train_encoder_output = bottleneck_features_train_dup.astype(np.float32)
test_encoder_output = bottleneck_features_test_dup.astype(np.float32)
validation_encoder_output = bottleneck_features_validation_dup.astype(np.float32)

test_decoder_input = np.argmax(test_decoder_input, axis=-1)
train_decoder_input = np.argmax(train_decoder_input, axis=-1)
validation_decoder_input = np.argmax(validation_decoder_input, axis=-1)

print("Decoder Input shape: {}, dtype: {}".format(train_decoder_input.shape, train_decoder_input.dtype))
print("Decoder Target shape: {}, dtype: {}".format(train_decoder_target.shape, train_decoder_target.dtype))
print("Encoder Output shape: {}, dtype: {}".format(train_encoder_output.shape, train_encoder_output.dtype))

print("Saving the final data to be used directly for training...")
np.savez('train_dev_test',
         train_encoder_output=train_encoder_output,
         train_decoder_input=train_decoder_input,
         train_decoder_target=train_decoder_target,
         validation_encoder_output=validation_encoder_output,
         validation_decoder_input=validation_decoder_input,
         validation_decoder_target=validation_decoder_target,
         test_encoder_output=test_encoder_output,
         test_decoder_input=test_decoder_input,
         test_decoder_target=test_decoder_target)
