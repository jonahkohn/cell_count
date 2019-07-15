import os
from tkinter import *
from tkinter import filedialog
from collections import OrderedDict
from datetime import date
import ast

import xmltodict
import tifffile
import numpy as np

from keras.models import load_model
from unet.metrics import recall, precision, f1, mcor
from unet.losses import weighted_bce_dice_loss
from unet import utils

from skimage.morphology import remove_small_objects, watershed
from skimage.measure import regionprops, label
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.filters import gaussian
import scipy.ndimage as ndi
from skimage.util import img_as_int

RED_PROBABILITY_CUTOFF = .75
RED_SMALL_OBJECT_CUTOFF = 25

GREEN_PROBABILITY_CUTOFF = .35
GREEN_SMALL_OBJECT_CUTOFF = 20

RED_MODEL_NAME = "red_model_v0.h5"
GREEN_MODEL_NAME = "green_model_v0.h5"

def init_model(model_path):
    """Loads the unet using the filepath specified, then returns a model object."""

    model = load_model(model_path,
                custom_objects={'recall': recall,
                                'precision': precision,
                                'f1': f1,
                                'mcor': mcor,
                                'weighted_bce_dice_loss': weighted_bce_dice_loss})
    return model


def load_images(save_structure):
    """Finds the MIP pathway, and returns the images loaded in tifffile."""

    image_paths = save_structure["image"]
    images = []
    for i in range(len(image_paths)):

        full_img = tifffile.imread(image_paths[i])
        img = resize(full_img, (1024, 1024, 1)) #converts from 4d np array to 3d
        images.append(img)

    return images


def segment(img, model, probability_cutoff = .80, small_object_cutoff = 30):
    """Uses a DeepFLaSH model to segment a 1024 x 1024 image, then binarizes at probability_cutoff. Watersheds and labels the image.

    model.predict() returns an array with shape (1, 1024, 1024, 1) in which every pixel's value is the model's confidence (from 0 to 1)
    that a cell exists in that pixel. This image is thresholded at the user-determined probability_cutoff. A watershed is performed,
    and then objects smaller than small_object_cutoff are removed. Each remaining object in the binary image is labelled.

    Paramaters
    ----------
    img : numpy array with shape of (1024, 1024). Single-channel image to be segmented.
    model : loaded DeepFLaSH model object.
    probability_cutoff : float from 0-1. From user-set global variable.
    small_object_cutoff : int passed to remove_small_objects. From user-set global variable.

    Returns
    -------
    image_labelled: numpy array in which every binary object shares a unique pixel value.
    r_props: list created by skimage.measure.regionprops(), linked to image_labelled.

    """

    image_predict = model.predict(np.asarray([img]))
    img = np.reshape(image_predict, (1024, 1024))

    image_predict = img

    cutoff_indexes = image_predict <= probability_cutoff
    image_predict[cutoff_indexes] = 0 #adjusts the image to only include pixels above probability_cutoff
    binary_indexes = image_predict > 0
    image_predict[binary_indexes] = 1 #sets all other pixels to 1

    image_predict = image_predict.astype(int)

    distance = ndi.distance_transform_edt(image_predict)
    distance_blur = gaussian(distance, sigma = 1.2, preserve_range = True)
    local_maxi = peak_local_max(distance_blur, indices = False, footprint = np.ones((8, 8)), labels = img)
    markers = ndi.label(local_maxi)[0]
    image_watershed = watershed(-distance_blur, markers, mask = image_predict, watershed_line = True)

    image_labelled = label(image_watershed)
    image_cleaned = remove_small_objects(image_labelled, small_object_cutoff, connectivity = 1, in_place = False)
    image_labelled, num = label(image_cleaned, return_num = True)

    print(str(num) + " labelled cells.")

    r_props = regionprops(image_labelled)

    return image_labelled, r_props


def count_overlap(labelled, props, double_cell_threshold = 85, cell_type_classifier = True):
    """Iterates over the red objects, and looks for overlapping green objects. Stores coordinates of overlap.

    Each red object's centroid location is accessed in the green labelled image. If the coordinates belong to an object in
    the green prop list, the green object's centroid location is accessed in the red image. If these coordinates refer to the
    original red object, the red centroid coordinates are appended to cell_coords.

    Optionally, various cell types can be marked by different properties. If cell_type_classifier is True, cell types
    other than "positive" will be recorded, and later stored as different cell markers in imageJ.

    Paramaters
    ----------
    labelled : tuple of red (0) and green (1) labelled images, as returned from segment().
    props: tuple of red (0) and green (1) regionprops, as returned from segment().
    cell_type_classifier : boolean, if True allows for the classification of cell types other than "positive."
    double_cell_threshold : int, threshold for "double" cell classification, referring to large cells.

    Returns
    -------
    cell_coords : A dictionary of lists, referring to cell classifications and their lists of recorded cell coordinates.

    """

    cell_coords = {"positive" : [],
                   "double" : [],
                   "red" : []
                   }

    red_props, green_props = props[0], props[1]

    red_labelled, green_labelled = labelled[0], labelled[1]

    for red_prop in red_props:

        red_centroid = red_prop.centroid  #coordinate tuple
        x, y = int(red_centroid[0]), int(red_centroid[1])

        red_centroid_label = green_labelled[x][y]   #int value of the red centroid coordinates on the green image

        if red_centroid_label != 0:
            g_prop = green_props[red_centroid_label - 1] #prop indexing starts at 1
            green_centroid = g_prop.centroid

            g_x, g_y = int(green_centroid[0]), int(green_centroid[1])

            green_centroid_label = red_labelled[g_x][g_y] #int value of the green centroid coordinates on the red image

            if green_centroid_label == red_prop.label:

                if red_prop.area > double_cell_threshold and cell_type_classifier:
                    cell_coords["double"].append((x, y))

                else:
                    cell_coords["positive"].append((x, y))

            elif cell_type_classifier:
                cell_coords["red"].append((x, y))

    return cell_coords


def save_counts(save_structure, cell_coords):
    """Writes all cell (x, y) coordinates into an xml file readable by the imageJ cellcount plugin.

    xmltodict is used to read and write xml files as dictionaries. All coordinates are offset by the amount
    recorded in save_structure["coords"], to account for the original cropping of the image. xml skeleton
    is found in main(), and read from save_structure["xml"].

    Paramaters
    ----------
    save_structure : dictionary containing the current iteration's directories.
    cell_coords : dictionary of lists containing cell coordinates and classifications, as returned from count_overlap.

    Returns
    -------
    None

    """

    with open(save_structure["xml"],'r') as f:
        cell_count_xml = xmltodict.parse(f.read())  #turns xml skeleton into dictionary

    with open(save_structure["coords"]) as f:
        offset = f.readline()
    offset = ast.literal_eval(offset) #reads the string as a tuple, for immediate typecasting.
    x_off, y_off = offset[0], offset[1]

    cell_counts = [[], [], []]

    for cell_type in cell_coords.keys():

        if cell_type == "positive":
            cell_marker_type = 0
        elif cell_type == "double":
            cell_marker_type = 1
        else:
            cell_marker_type = 2

        if len(cell_coords[cell_type]) != 0:

            for cell in cell_coords[cell_type]:

                x, y = cell[0], cell[1]

                marker_dict = OrderedDict()
                marker_dict['MarkerX'] = y + x_off #offset added to account for cropping
                marker_dict['MarkerY'] = x + y_off  #coordinates in XML counter are flipped !?!?!?!?
                marker_dict['MarkerZ'] = 1

                cell_counts[cell_marker_type].append(marker_dict)

            cell_count_xml['CellCounter_Marker_File']["Marker_Data"]['Marker_Type'][cell_marker_type]["Marker"] = cell_counts[cell_marker_type]

            cell_count_xml['CellCounter_Marker_File']["Image_Properties"]["Image_Filename"] = "single_composite_MIP.tif"

    save_xml = os.path.join(save_structure["save"],"single_cell_count_auto.xml")
    xml_string = xmltodict.unparse(cell_count_xml, pretty=True, newl="\n",indent="  ")

    with open(save_xml,'w') as f:
        f.write(xml_string)

    save_red_label = os.path.join(save_structure["save"],"single_red_labelled_mip.tif")
    save_green_label = os.path.join(save_structure["save"],"single_green_labelled_mip.tif")

    red_label = save_structure["labelled"][0]
    green_label = save_structure["labelled"][1]

    tifffile.imsave(save_red_label, red_label.astype(np.uint16))
    tifffile.imsave(save_green_label, green_label.astype(np.uint16))

    mip_directory = os.path.dirname(save_structure["save"])
    fileset = []
    for files in os.listdir(mip_directory):

        if files.endswith('tif'):
            fileset.append(os.path.join(mip_directory, files))

    fileset = fileset[::-1]
    images = []
    for tif in fileset:
        images.append(tifffile.imread(tif))

    images = np.asarray(images)

    finalfilename = os.path.join(save_structure["save"], "single_composite_MIP.tif")
    tifffile.imsave(finalfilename, images)

def main():
    """

    Counts the cells for a single MIP image folder, and saves the counted cells into an xml file in the composite folder.

    Finds the MIP images and then loads and segments them using load_images() and segment(). Counts the cells and saves
    the coordinates and segmented images using count_overlap(), save_labelled(), and save_counts(). Assumes the existence
    of a cc_save_data folder containing ch01 and ch02 cropped images, as well as crop coordinates in a .txt file.

    Paramaters
    ----------
    None

    Returns
    -------
    None

    """

    cwd = os.getcwd()

    basicxml = [os.path.join(cwd,f) for f in os.listdir(cwd) if 'basicxml' in f][0] #skeleton file for imagej plugin

    root = Tk()
    root.withdraw()
    save_directory = filedialog.askdirectory(initialdir=cwd,title='Select the cropped MIP folder')

    #finds ch01 and ch02 mip files
    MIP_locations = sorted([os.path.join(save_directory,f) for f in os.listdir(save_directory) if (f.endswith('tif') and ("label" not in f))])

    #finds crop location to offset cell count markers
    crop_coord_location = [os.path.join(save_directory,f) for f in os.listdir(save_directory) if (f.endswith('coords.txt'))][0]

    red_model_path = os.path.join(RED_MODEL_NAME)
    green_model_path = os.path.join(GREEN_MODEL_NAME)
    red_model = init_model(red_model_path)
    green_model = init_model(green_model_path)

    save_structure = {"cwd" : cwd,
                      "image" : MIP_locations,
                      "save" : save_directory,
                      "models" : (red_model, green_model),
                      "xml" : basicxml,
                      "coords" : crop_coord_location
                      }

    mip_images = load_images(save_structure)
    print("----------------------------------------------------------")
    print("Loaded images and model.")

    r_labelled, r_props = segment(mip_images[1], red_model, probability_cutoff = RED_PROBABILITY_CUTOFF, small_object_cutoff = RED_SMALL_OBJECT_CUTOFF)
    g_labelled, g_props = segment(mip_images[0], green_model, probability_cutoff = GREEN_PROBABILITY_CUTOFF, small_object_cutoff = GREEN_SMALL_OBJECT_CUTOFF)
    print("Segmented images.")

    props = [r_props, g_props]
    labelled = [r_labelled, g_labelled]
    save_structure["labelled"] = labelled

    cell_coords = count_overlap(labelled, props, double_cell_threshold = 100, cell_type_classifier = False)
    print("Counted cells.")

    save_counts(save_structure, cell_coords)
    print("Saved, exiting.")

if __name__ == "__main__":
    main()
