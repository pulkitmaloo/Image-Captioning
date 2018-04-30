from copy import deepcopy
from heapq import nsmallest
from caption_utils import *
from tqdm import tqdm
from keras.models import load_model
from keras.preprocessing import image
from argparse import ArgumentParser
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


def seq_to_sentence(sent):
    return ' '.join([idx2token[idx] for idx in sent])


def generate_seq(img_input, alpha=1.):
    if img_input.shape != (1, 512):
        img_input = img_input.reshape(1, 512)
    
    assert(img_input.shape == (1, 512))
    stop_condition = False
    decoded_sentence = []
    target_seq = np.array([token2idx['<bos>']]).reshape(1, 1)
    states_value = encoder_model.predict(img_input)
    
    neg_log_proba = 0.
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        output_tokens = np.squeeze(output_tokens)
        
        sampled_token_index = int(np.argmax(output_tokens))
        neg_log_proba -= np.log(output_tokens[sampled_token_index])
        
        sampled_char = idx2token[sampled_token_index]

        decoded_sentence += [sampled_char]

        if (sampled_char == '<eos>' or len(decoded_sentence) > 30):
            stop_condition = True

        target_seq = np.array([sampled_token_index]).reshape(1, 1)

        states_value = [h, c]
        neg_log_proba /= len(decoded_sentence)**alpha
    return ' '.join(decoded_sentence[: -1])


def decoder_one_step(sent, beam_size=5, len_norm=True, alpha=1):
    """ 
    sent: ([neg_log_prob, [1, ...]], [h, c])
    states_value: [h, c]
    return list of sent
    """
    prev_log_prob = sent[0][0]
    prev_sent = sent[0][1]
    last_word_idx = prev_sent[-1]
    states_value = sent[1] 
    
    assert last_word_idx not in (token2idx['<eos>'], token2idx['<unk>']) 
    
    target_seq = np.array([last_word_idx]).reshape(1, 1)
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    output_tokens = np.squeeze(output_tokens)
    
    predicted_sentences = []    
#     output_tokens_beam = np.argpartition(-output_tokens, beam_size+4)
    output_tokens_beam = np.argsort(-output_tokens)
    output_tokens_beam = list(filter(lambda x: x not in [0, 1, 3], output_tokens_beam))[: beam_size]
    
    assert len(output_tokens_beam) == beam_size
    
    for predict_idx in output_tokens_beam:
#         if predict_idx in [0, 1, 3]:
#             continue
        
        new_sent = prev_sent + [int(predict_idx)]                
        
        if len_norm:
            neg_log_prob = prev_log_prob * max(len(prev_sent)-1, 1)**alpha - np.log(output_tokens[int(predict_idx)])
            neg_log_prob /= max(len(new_sent)-1, 1)**alpha
        else:
            neg_log_prob = prev_log_prob - np.log(output_tokens[int(predict_idx)])
            
        predicted_sentences.append(([neg_log_prob, new_sent], [h, c]))
        
#     print("from", sent[0][0], seq_to_sentence(sent[0][1]))
#     print("predicting")
#     for s in predicted_sentences:
#         print(s[0][0], seq_to_sentence(s[0][1]))
    
    return predicted_sentences
    
    
def beam_search(img_input, beam_size=5, max_length=20, len_norm=True, alpha=1.):
    """throws an error on beam_size 1 when <unk> is produced"""
    if img_input.shape != (1, 512):
        img_input = img_input.reshape(1, 512)    
    assert(img_input.shape == (1, 512))
    states_value_initial = encoder_model.predict(img_input)
    
    beg_sent_and_states = ([0., [token2idx['<bos>']]], states_value_initial)
#     print(beg_sent)
    top_sentences = decoder_one_step(beg_sent_and_states, beam_size, len_norm, alpha)
#     print(list(map(lambda x: seq_to_sentence(x[1]), top_sentences)))
    
    stop_condition = False
    
    while not stop_condition:
        new_top_sentences = []
        for sent in top_sentences:
            if sent[0][1][-1] == token2idx['<eos>']:
                new_top_sentences.append(sent)
                continue
                
            predicted_sent = decoder_one_step(sent, beam_size, len_norm, alpha)
            new_top_sentences.extend(predicted_sent)
            
        top_sentences = sorted(new_top_sentences, key=lambda x: x[0][0])[: beam_size]
        assert len(top_sentences) == beam_size

#         print(seq_to_sentence(top_sentences[0][1]))
        
        # Update stop condition
        eos_cnt = 0
        any_max_len = False
        for sent in top_sentences:
            if sent[0][1][-1] == token2idx['<eos>']:
                eos_cnt += 1
            if len(sent[0][1]) >= max_length:
                any_max_len = True
                #print('Max len reached')
                break
        
        if any_max_len or (eos_cnt == beam_size):
            stop_condition = True        
            
    return list(map(lambda x: str(round(x[0][0], 2)) +' '+ seq_to_sentence(x[0][1][1: -1]), top_sentences))


def get_image_features(img_path):
    VGG16_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    #img_path = 'data/Arnav_Hankyu_Pulkit2.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = VGG16_model.predict(x)
    return features


def visualize_example(img_fname, captions):
    img = image.load_img(img_fname, target_size=(224, 224))
    #plt.imread("data/Flicker8k_Dataset/" + img_fname, )
    plt.title("\n".join(captions))
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.imshow(img)
    plt.savefig("results/" + img_fname)
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Image Captioning")
    parser.add_argument('-f', '--file_name', type=str, default=None, 
                        help="File Name, None for running on test images of Flikr8k Dataset")
    parser.add_argument('-em','--encoder_model', type=str, default='saved_models/encoder_model.h5', 
                        help ="File path for the encoder model")
    parser.add_argument('-dm','--decoder_model', type=str, default='saved_models/decoder_model.h5', 
                        help ="File path for the decoder model")
    parser.add_argument('-bs', '--beam_size', type=int, default=5, help="Beam Size")
    parser.add_argument('-l', '--max_length', type=int, default=20, help="Max Length of the generated sentences")
    parser.add_argument('-ln', '--length_normalization', type=bool, default=True, help="Length Normalization")
    parser.add_argument('-a', '--alpha', type=float, default=0.7, help="Alpha for length normalization")
    
    args = parser.parse_args()
    beam_size = args.beam_size
    max_length = args.max_length
    len_norm = args.length_normalization
    alpha = args.alpha
    file_name = args.file_name
    encoder_model = args.encoder_model
    decoder_model = args.decoder_model
    
    train_fns_list, dev_fns_list, test_fns_list = load_split_lists()
    train_captions_raw, dev_captions_raw, test_captions_raw = get_caption_split()
    vocab = create_vocab(train_captions_raw)
    token2idx, idx2token = vocab_to_index(vocab)    
    captions_data = (train_captions_raw.copy(), dev_captions_raw.copy(), test_captions_raw.copy())
    train_captions, dev_captions, test_captions = process_captions(captions_data, token2idx)
    
    encoder_model = load_model(encoder_model)
    decoder_model = load_model(decoder_model)
    
    if file_name:
        img_input = get_image_features(file_name)
        generated_captions = beam_search(img_input, beam_size=beam_size, max_length=max_length, len_norm=len_norm, alpha=alpha)
        visualize_example(file_name, generated_captions)
        print('\n'.join(generated_captions))
        
    else:        
        all_data = np.load('train_dev_test.npz')
        test_encoder_output = all_data['test_encoder_output']
        test_decoder_target = all_data['test_decoder_target'][:,1:,:]    

        for i, fname in tqdm(enumerate(test_fns_list)):
            img_input = test_encoder_output[i*5, :]
            
            generated_captions = beam_search(img_input, beam_size=beam_size, max_length=max_length, len_norm=len_norm, alpha=alpha)
    #        original_caption = seq_to_sentence(np.argmax(test_decoder_target[i, :], -1))
    #        original_caption = original_caption[: original_caption.index('<')]
    
            visualize_example("data/Flicker8k_Dataset/" + fname, generated_captions)
