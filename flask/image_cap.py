from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import json



# def generate_seq(img_input):
    
    # with open("token2idx.json",'r') as fp:
        # dict1 = json.load(fp)

    # with open("idx2token.json",'r') as fp:
        # dict2 = json.load(fp)

    # token2idx = {k:int(v) for k,v in dict1.items()}
    # idx2token = {int(k):v for k,v in dict2.items()}
    # if img_input.shape != (1, 512):
        # img_input = img_input.reshape(1, 512)

    # encoder_model = load_model('../saved_models/encoder_model.h5')
    # decoder_model = load_model('../saved_models/decoder_model.h5')
    # assert(img_input.shape == (1, 512))
    # stop_condition = False
    # decoded_sentence = []
    # target_seq = np.array([token2idx['<bos>']]).reshape(1, 1)
    # states_value = encoder_model.predict(img_input)

    # while not stop_condition:
        # output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        # sampled_char = idx2token[sampled_token_index]
        # decoded_sentence += [sampled_char]
        # if (sampled_char == '<eos>' or len(decoded_sentence) > 30):
            # stop_condition = True
        # target_seq = np.array([sampled_token_index]).reshape(1, 1)
        # states_value = [h, c]

    # return ' '.join(decoded_sentence[:-1])

# def get_captions(img_path):

    # VGG16_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    # #img_path = 'data/Arnav_Hankyu_Pulkit2.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    # features = VGG16_model.predict(x)
    # return generate_seq(features)

##############################################################################


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


def decoder_one_step(sent, decoder_model, beam_size=5, len_norm=True, alpha=1):
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
    
    
def beam_search(img_input, encoder_model, decoder_model, input_shape=512, beam_size=5, max_length=20, len_norm=True, alpha=1., return_probs=False):
    """throws an error on beam_size 1 when <unk> is produced"""
    if img_input.shape != (1, input_shape):
        img_input = img_input.reshape(1, input_shape)    
    assert(img_input.shape == (1, input_shape))
    states_value_initial = encoder_model.predict(img_input)
    
    with open("token2idx.json",'r') as fp:
        dict1 = json.load(fp)
    global token2idx 
    token2idx = {k:int(v) for k,v in dict1.items()}

    beg_sent_and_states = ([0., [token2idx['<bos>']]], states_value_initial)
#     print(beg_sent)
    top_sentences = decoder_one_step(beg_sent_and_states, decoder_model, beam_size, len_norm, alpha)
#     print(list(map(lambda x: seq_to_sentence(x[1]), top_sentences)))
    
    stop_condition = False
    
    while not stop_condition:
        new_top_sentences = []
        for sent in top_sentences:
            if sent[0][1][-1] == token2idx['<eos>']:
                new_top_sentences.append(sent)
                continue
                
            predicted_sent = decoder_one_step(sent, decoder_model, beam_size, len_norm, alpha)
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
           
    if return_probs:
        return list(map(lambda x: str(round(x[0][0], 2)) +' '+ seq_to_sentence(x[0][1][1: -1]), top_sentences))
    else:
        return list(map(lambda x: seq_to_sentence(x[0][1][1: -1]), top_sentences))[0]

def get_image_features(img_path, model):
    if model == "VGG16":
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input
        pretrained_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    elif model == "VGG19":
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input
        pretrained_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    elif model == "ResNet50":
        from keras.applications.resnet50 import ResNet50
        from keras.applications.resnet50 import preprocess_input
        pretrained_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
      
      
    #img_path = 'data/Arnav_Hankyu_Pulkit2.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = pretrained_model.predict(x)
    return features


def get_captions(img_path, model='ResNet50'):

    with open("token2idx.json",'r') as fp:
        dict1 = json.load(fp)
    global token2idx 
    token2idx = {k:int(v) for k,v in dict1.items()}

    with open("idx2token.json",'r') as fp:
        dict2 = json.load(fp)
    global idx2token 
    idx2token = {int(k):v for k,v in dict2.items()}

    if model == 'VGG16':
        input_shape = 512
    if model == 'VGG19':
        input_shape = 512
    elif model == 'ResNet50':
        input_shape = 2048
    else:
        print('Model not found')

    # define model 
    # define encoder model
    encoder_model = load_model('../saved_models/encoder_model_ResNet50_lr000051_emb512.h5')
    decoder_model = load_model('../saved_models/decoder_model_ResNet50_lr000051_emb512.h5')
    beam_size = 5
    max_length = 20
    len_norm=True
    alpha=0.7
    return_probs=False

    img_input = get_image_features(img_path, model)

    generated_captions = beam_search(img_input, encoder_model=encoder_model, decoder_model=decoder_model, beam_size=beam_size, max_length=max_length, len_norm=len_norm, alpha=alpha, return_probs=return_probs, input_shape=input_shape)
    return generated_captions


if __name__ == "__main__":
    print(get_captions("../data/test_image.png"))
