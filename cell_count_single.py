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

    model = load_model(model_path,
                custom_objects={'recall': recall,
                                'precision': precision,
                                'f1': f1,
                                'mcor': mcor,
                                'weighted_bce_dice_loss': weighted_bce_dice_loss})
    return model


def load_images(save_structure):

    image_paths = save_structure["image"]
    images = []
    for i in range(len(image_paths)):

        full_img = tifffile.imread(image_paths[i])
        img = resize(full_img, (1024, 1024, 1)) #converts from 4d np array to 3d
        images.append(img)

    return images


# takes an image and a ML model, and runs prediction and image cleaanup.
# returns the modified image and a list of skimage regionprop objects.
def segment(img, model, probability_cutoff = .80, small_object_cutoff = 30):

    image_predict = model.predict(np.asarray([img]))
    image_predict = np.reshape(image_predict, (1024, 1024))

    cutoff_indexes = image_predict <= probability_cutoff
    image_predict[cutoff_indexes] = 0 #adjusts the image to only include pixels above probability_cutoff
    binary_indexes = image_predict > 0
    image_predict[binary_indexes] = 1 #sets all other pixels to 1

    image_predict = image_predict.astype(int)

    distance = ndi.distance_transform_edt(image_predict)
    local_maxi = peak_local_max(distance, min_distance=15, indices=False, footprint=np.ones((10, 10)), labels=image_predict)
    markers = ndi.label(local_maxi)[0]
    image_watershed = watershed(-distance, markers, mask=image_predict, compactness = 0.9, watershed_line = True)

    image_labelled = label(image_watershed)
    image_cleaned = remove_small_objects(image_labelled, small_object_cutoff, connectivity=2, in_place = False)
    image_labelled = label(image_cleaned)

    r_props = regionprops(image_labelled)

    return image_labelled, r_props


# takes red and green labelled images (in list) and props (in list) and determines whether cells overlap.
# centroids (regionprops) are compared to the opposite-channel's labelled image.
# returns a dictionary of cell classifications, which refer to a list of coordinates.
def count_overlap(labelled, props, double_cell_threshold = 85, cell_type_classifier = True):

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

#converts cell_coords into an xml file formatted for the cell counting imageJ plugin, and saves the labelled
#red and green images to the designated folder.
def save_counts(save_structure, cell_coords):

    with open(save_structure["xml"],'r') as f:
        cell_count_xml = xmltodict.parse(f.read())  #turns xml skeleton into dictionary

    with open(save_structure["coords"]) as f:
        offset = f.readline()
    offset = ast.literal_eval(offset)
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
                marker_dict['MarkerX'] = y + x_off
                marker_dict['MarkerY'] = x + y_off  #coordinates in XML counter are flipped !?!?!?!?
                marker_dict['MarkerZ'] = 1

                cell_counts[cell_marker_type].append(marker_dict)

            cell_count_xml['CellCounter_Marker_File']["Marker_Data"]['Marker_Type'][cell_marker_type]["Marker"] = cell_counts[cell_marker_type]

            cell_count_xml['CellCounter_Marker_File']["Image_Properties"]["Image_Filename"] = "DualLabeled_MIP.tif"

    save_xml = os.path.join(save_structure["save"],"cell_count_auto.xml")
    xml_string = xmltodict.unparse(cell_count_xml, pretty=True, newl="\n",indent="  ")

    with open(save_xml,'w') as f:
        f.write(xml_string)

    save_red_label = os.path.join(save_structure["save"],"red_labelled_mip.tif")
    save_green_label = os.path.join(save_structure["save"],"green_labelled_mip.tif")

    red_label = save_structure["labelled"][0]
    green_label = save_structure["labelled"][1]

    tifffile.imsave(save_red_label, red_label.astype(np.uint16))
    tifffile.imsave(save_green_label, green_label.astype(np.uint16))

    mippath = os.path.dirname(save_structure["save"])
    saveMIP = os.path.join(save_structure["save"], "DualLabeled_MIP.tif")

    dual_dir = sorted([os.path.join(mippath, f) for f in os.listdir(mippath) if (f.endswith('tif') and ('ch00' not in f))])

    dualmip = []
    for f in dual_dir:
        dualmip.append(tifffile.imread(f))
    dualmip = np.asarray(dualmip)
    tifffile.imsave(saveMIP, dualmip)


"""
Main
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
