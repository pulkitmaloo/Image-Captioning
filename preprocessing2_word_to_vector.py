from keras.utils import to_categorical
from caption_utils import *
from tqdm import tqdm


def one_hot_encode(fns_list, caption_dictionary, num_captions_per_image, max_words_in_sentence, num_words):
    total_captions = len(caption_dictionary) * num_captions_per_image
    captions_onehot_processed = np.zeros((total_captions, max_words_in_sentence, num_words)).astype(bool)
    for i, filename in tqdm(enumerate(fns_list)):
        for j, caption in enumerate(caption_dictionary[filename]):
            onehot = to_categorical(caption, num_classes=len(vocab))
            for k, word in enumerate(onehot):
                captions_onehot_processed[i*num_captions_per_image+j][k] = word
        # if (i%100 == 99):
            # print("{} / {} images processed".format(i+1, len(caption_dictionary)))
    return captions_onehot_processed


train_fns_list, validation_fns_list, test_fns_list = load_split_lists()

train_captions_raw, validation_captions_raw, test_captions_raw = get_caption_split()
vocab = create_vocab(train_captions_raw)
token2idx, idx2token = vocab_to_index(vocab)
captions_data = (train_captions_raw.copy(), validation_captions_raw.copy(), test_captions_raw.copy())
train_captions, validation_captions, test_captions = process_captions(captions_data, token2idx)

print("\nTotal vocabularies in our dictionary: {}".format(len(vocab)))

# Calculate the caption with maximum number of words
caption_lengths = []
for five_captions in train_captions.values():
    caption_lengths.extend([len(caption) for caption in five_captions]) 
for five_captions in test_captions.values():
    caption_lengths.extend([len(caption) for caption in five_captions]) 
for five_captions in validation_captions.values():
    caption_lengths.extend([len(caption) for caption in five_captions]) 

max_words_in_sentence = max(caption_lengths)

print("\nThere are {} number of captions in total.".format(len(caption_lengths)))
print("The maximum words in a sentence is {}".format(max_words_in_sentence))

# Save train captions
num_words = len(vocab)
num_captions_per_image = 5 # 5 stands for number of captions per image

print("\nGenerating onehot vectors for train captions...")
train_captions_onehot_processed = one_hot_encode(train_fns_list, train_captions, num_captions_per_image, max_words_in_sentence, num_words)

print("\nGenerating onehot vectors for test captions...")
test_captions_onehot_processed = one_hot_encode(test_fns_list, test_captions, num_captions_per_image, max_words_in_sentence, num_words)

print("\nGenerating onehot vectors for validation captions...")
validation_captions_onehot_processed = one_hot_encode(validation_fns_list, validation_captions, num_captions_per_image, max_words_in_sentence, num_words)

print("\nSaving the preprocessed captions...")
np.savez('preprocessed_captions/Flicker8k_onehot_' + str(len(vocab)) + '_words',
        train=train_captions_onehot_processed,
        test=test_captions_onehot_processed,
        validation=validation_captions_onehot_processed)
