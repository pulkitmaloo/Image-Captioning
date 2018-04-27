import glob
import numpy as np
from keras.preprocessing import image

#############################################################
# Things to do before running the script
#############################################################
# Place 'Flicker8k_Dataset' in the directory with images
# Create bottleneck_features directory
#############################################################
def get_filenames(txtfile):
    with open(txtfile) as f:
        file_paths = []
        for line in f:
            file_paths.append('data/Flicker8k_Dataset/' + line.rstrip())
    return file_paths

def generate_bottleneck_features(model, dim_last_layer, filename_list):
    bottleneck_features = np.zeros((len(filename_list), dim_last_layer))
    for i, path in enumerate(filename_list):
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        bottleneck_features[i] = features
    return bottleneck_features

# Initialize bottleneck_features
train_files = get_filenames('data/Flickr8k_text/Flickr_8k.trainImages.txt')
test_files = get_filenames('data/Flickr8k_text/Flickr_8k.testImages.txt')
validation_files = get_filenames('data/Flickr8k_text/Flickr_8k.devImages.txt')

print("Number of train files: {}".format(len(train_files)))
print("Number of test files: {}".format(len(test_files)))
print("Number of validation files: {}".format(len(validation_files)))

#############################################################
# VGG16 (last layer: 512)
#############################################################
print("\nGenerating bottleneck features for VGG16 ...")

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False, pooling='avg')

bottleneck_features_train = generate_bottleneck_features(model, 512, train_files)
bottleneck_features_test = generate_bottleneck_features(model, 512, test_files)
bottleneck_features_validation = generate_bottleneck_features(model, 512, validation_files)

print("\nSaving bottleneck features for VGG16 ...")
np.savez('bottleneck_features/Flicker8k_bottleneck_features_VGG16_avgpooling',
         train=bottleneck_features_train,
         test=bottleneck_features_test,
         validation=bottleneck_features_validation)

#############################################################
# VGG19 (last layer: 512)
#############################################################
print("\nGenerating bottleneck features for VGG19 ...")

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

model = VGG19(weights='imagenet', include_top=False, pooling='avg')

bottleneck_features_train = generate_bottleneck_features(model, 512, train_files)
bottleneck_features_test = generate_bottleneck_features(model, 512, test_files)
bottleneck_features_validation = generate_bottleneck_features(model, 512, validation_files)

print("\nSaving bottleneck features for VGG19 ...")
np.savez('bottleneck_features/Flicker8k_bottleneck_features_VGG19_avgpooling',
         train=bottleneck_features_train,
         test=bottleneck_features_test,
         validation=bottleneck_features_validation)

#############################################################
# ResNet50 (last layer: 2048)
#############################################################
print("\nGenerating bottleneck features for ResNet50 ...")

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

bottleneck_features_train = generate_bottleneck_features(model, 2048, train_files)
bottleneck_features_test = generate_bottleneck_features(model, 2048, test_files)
bottleneck_features_validation = generate_bottleneck_features(model, 2048, validation_files)

print("\nSaving bottleneck features for ResNet50 ...")
np.savez('bottleneck_features/Flicker8k_bottleneck_features_ResNet50_avgpooling',
         train=bottleneck_features_train,
         test=bottleneck_features_test,
         validation=bottleneck_features_validation)
