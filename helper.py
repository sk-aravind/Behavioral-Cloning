import cv2
import numpy as np

###================== Helper Functions ==================###

def get_file_name(full_path):
    return ("./data/IMG/" + full_path.split('/')[-1])

###================== Augmentation Functions ==================###

def mirror(feature, label=None):
    '''
    Returns flipped images and flipped steering angle
    '''
    if label == None:
        feature = cv2.flip(feature, 1)
        return feature
    else:
        feature = cv2.flip(feature, 1)
        label *= -1.0
    return feature, label

def get_random_translation(maximum_translation):
    '''
    Returns random translation within bounds of +-maximum_translation
    '''
    return maximum_translation*np.random.uniform() - maximum_translation/2

def random_affine_transformation(image, angle, shear_range=200):
    """
    Return randomly transformed image with scaled steering angle
    """
    rows, cols = image.shape[0:2]
    dx = np.random.randint(-shear_range, shear_range)
    random_point = [cols/2 + dx, rows/2]
    triangle1 = np.float32([[0,         rows],
                            [cols,      rows],
                            [cols/2,    rows/2]])
    triangle2 = np.float32([[0,    rows],
                            [cols, rows],
                            random_point])

    steering_correction = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    transf_matrix = cv2.getAffineTransform(triangle1, triangle2)
    image = cv2.warpAffine(image, transf_matrix, (cols, rows), borderMode=1)
    angle += steering_correction

    return image, angle

def translate(feature, label=None):
    '''
    Returns translated image with proportionally scaled steering angle
    '''
    if label == None:
        horizontal_translation = get_random_translation(maximum_translation=100)
        vertical_translation = 0

        transformation_matrix = np.float32([[1, 0, horizontal_translation], [0, 1, vertical_translation]])
        output_size = (320, 160)
        feature = cv2.warpAffine(feature, transformation_matrix, output_size)
        return feature

    else:
        horizontal_translation = get_random_translation(maximum_translation=110)
        vertical_translation = get_random_translation(maximum_translation=40)

        transformation_matrix = np.float32([[1, 0, horizontal_translation], [0, 1, vertical_translation]])
        output_size = (320, 160)
        feature = cv2.warpAffine(feature, transformation_matrix, output_size)
        label += horizontal_translation * 0.004  # we add 0.004 steering angle for every translated pixel
    return feature, label

def augment_data(feature,label):
    '''
    Applies one of the augmentations and returns img with steering label
    Percentage of each augmentation is controlled by the limits in the if statement
    This allows us to easily tweak our dataset to fix problems in driving behaviours
    '''
    if np.random.normal(0.5, 0.5) < 0.8 :
        feature,label = mirror(feature,label)
    if np.random.normal(0.5, 0.5) < 0.65:
        feature,label = translate(feature,label)
    else:
        feature,label = random_affine_transformation(feature,label)

    return feature,label
