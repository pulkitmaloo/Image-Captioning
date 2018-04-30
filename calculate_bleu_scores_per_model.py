from argparse import ArgumentParser
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

from caption_utils import *
from inference import *


def generate_seq(img_input, input_shape, encoder_model, decoder_model):

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

    return ' '.join(decoded_sentence[:-1])

def get_captions(model, img_path, input_shape, encoder_model, decoder_model):   
    #img_path = 'data/Arnav_Hankyu_Pulkit2.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    return beam_search(features, input_shape=input_shape, encoder_model=encoder_model, decoder_model=decoder_model)

def get_reference_and_candidates(model, fns_list, input_shape, encoder_model, decoder_model):
    all_refs = []
    all_candidates = []

    for i, filename in enumerate(fns_list):
        if i%100 == 0:
            print("{} / {} images processed".format(i, len(fns_list)))
        candidate = get_captions(model, "data/Flicker8k_Dataset/"+filename, input_shape, encoder_model, decoder_model).split()
        references = []
        for j, caption in enumerate(test_captions_raw[filename]):
            references.append(caption[:-1].split())
        all_refs.append(references)
        all_candidates.append(candidate)
    return all_refs, all_candidates

def calculate_bleu_scores(all_refs, all_candidates):
    bleu1 = corpus_bleu(all_refs, all_candidates, weights=(1, 0, 0, 0)) * 100
    bleu2 = corpus_bleu(all_refs, all_candidates, weights=(0, 1, 0, 0)) * 100
    bleu3 = corpus_bleu(all_refs, all_candidates, weights=(0, 0, 1, 0)) * 100
    bleu4 = corpus_bleu(all_refs, all_candidates, weights=(0, 0, 0, 1)) * 100
    return bleu1, bleu2, bleu3, bleu4

if __name__ == "__main__":
    parser = ArgumentParser(description="Image Captioning")
    parser.add_argument('-m', '--model', type=str, default="VGG16", help="Pretrained model for images")
    parser.add_argument('-tm', '--trained_model', type=str, default="test.h5", help="filename to save the trained model")
    parser.add_argument('-em', '--encoder_model', type=str, default="encoder_model.h5", help="filename to save the encoder model")
    parser.add_argument('-dm', '--decoder_model', type=str, default="decoder_model.h5", help="filename to save the decoder model")

    args = parser.parse_args()
    model = args.model
    trained_model = args.trained_model
    encoder_model = args.encoder_model
    decoder_model = args.decoder_model

    train_fns_list, dev_fns_list, test_fns_list = load_split_lists()
    train_captions_raw, dev_captions_raw, test_captions_raw = get_caption_split()
    vocab = create_vocab(train_captions_raw)
    token2idx, idx2token = vocab_to_index(vocab)

    if model == "VGG16":
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input
        pretrained_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        input_shape = 512
    elif model == "VGG19":
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input
        pretrained_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
        input_shape = 512
    elif model == "ResNet50":
        from keras.applications.resnet50 import ResNet50
        from keras.applications.resnet50 import preprocess_input
        pretrained_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        input_shape = 2048

    print("\nLoading models...")
    encoder_model = load_model(encoder_model)
    decoder_model = load_model(decoder_model)

    print("\nCalculating bleu scores...")
    all_refs, all_candidates = get_reference_and_candidates(pretrained_model, test_fns_list, input_shape, encoder_model, decoder_model)
    bleu_scores = calculate_bleu_scores(all_refs, all_candidates)

    for bleu_score in bleu_scores:
        print("Bleu1: {:0.2f}".format(bleu_score))
