from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np
import json

def generate_seq(img_input, model):
    
    with open("token2idx.json",'r') as fp:
        token2idx = json.load(fp)
        
    with open("idx2token.json",'r') as fp:
        idx2token = json.load(fp)
    
    if img_input.shape != (1, 512):
        img_input = img_input.reshape(1, 512)
    
    assert(img_input.shape == (1, 512))
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


def get_captions(img_path):

    VGG16_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    model  = load_model("../test.h5")
    
    #img_path = 'data/Arnav_Hankyu_Pulkit2.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = VGG16_model.predict(x)
    return generate_seq(features, model)

if __name__ == "__main__":
    print(get_captions("../data/test_image.png"))
