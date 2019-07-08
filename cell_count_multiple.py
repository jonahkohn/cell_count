import numpy as np
import os
import tifffile
from tkinter import *
from tkinter import filedialog
from collections import OrderedDict
from datetime import date

import ast
import xmltodict
import dicttoxml

from keras.models import load_model
from unet.metrics import recall, precision, f1, mcor
from unet.losses import weighted_bce_dice_loss
from unet import utils

from skimage.morphology import remove_small_objects, watershed
from skimage.measure import regionprops, label
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.util import img_as_int
import scipy.ndimage as ndi

RED_PROBABILITY_CUTOFF = .55
RED_SMALL_OBJECT_CUTOFF = 20

GREEN_PROBABILITY_CUTOFF = .3
GREEN_SMALL_OBJECT_CUTOFF = 10

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
def segment(img, model, probability_cutoff = .50, small_object_cutoff = 30):

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
    image_cleaned = remove_small_objects(image_labelled, small_object_cutoff, connectivity=1, in_place = False)
    image_labelled, num = label(image_cleaned, return_num = True)

    print("Num Labelled: " + str(num))

    r_props = regionprops(image_labelled)

    return image_labelled, r_props


# takes red and green labelled images (in list) and props (in list) and determines whether cells overlap.
# centroids (regionprops) are compared to the opposite-channel's labelled image.
# returns a dictionary of cell classifications, which refer to a list of coordinates.
def count_overlap(labelled, props, double_cell_threshold = 85, cell_type_classifier = False):

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


#converts cell_coords into an xml file formatted for the cell counting imageJ plugin.
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

            composite = os.path.basename(save_structure["composite"])

            cell_count_xml['CellCounter_Marker_File']["Image_Properties"]["Image_Filename"] = composite

    xml_id = save_structure["composite"].replace(".tif", "_count.xml")

    save_xml = os.path.join(save_structure["save"], xml_id)
    xml_string = xmltodict.unparse(cell_count_xml, pretty=True, newl="\n",indent="  ")

    with open(save_xml,'w') as f:
        f.write(xml_string)


def save_mips(save_structure):

    base = os.path.basename(save_structure["composite"])

    red_filename = base.replace(".tif", "_red_label.tif")
    green_filename = base.replace(".tif", "_green_label.tif")

    save_red_label = os.path.join(save_structure["save"], red_filename)
    save_green_label = os.path.join(save_structure["save"], green_filename)

    red_label = save_structure["labelled"][0]
    green_label = save_structure["labelled"][1]

    tifffile.imsave(save_red_label, red_label.astype(np.uint16))
    tifffile.imsave(save_green_label, green_label.astype(np.uint16))


def run(save_structure):

    crop_dir = save_structure["save"]

    MIP_locations = sorted([os.path.join(crop_dir,f) for f in os.listdir(crop_dir) if (f.endswith('tif') and ("label" not in f))])
    crop_coord_location = [os.path.join(crop_dir,f) for f in os.listdir(crop_dir) if (f.endswith('coords.txt'))][0]

    save_structure["image"] =  MIP_locations
    save_structure["coords"] = crop_coord_location

    mip_images = load_images(save_structure)

    r_labelled, r_props = segment(mip_images[1], save_structure["models"][0], probability_cutoff = RED_PROBABILITY_CUTOFF, small_object_cutoff = RED_SMALL_OBJECT_CUTOFF)
    g_labelled, g_props = segment(mip_images[0], save_structure["models"][1], probability_cutoff = GREEN_PROBABILITY_CUTOFF, small_object_cutoff = GREEN_SMALL_OBJECT_CUTOFF)

    props = [r_props, g_props]
    labelled = [r_labelled, g_labelled]
    save_structure["labelled"] = labelled

    cell_coords = count_overlap(labelled, props, double_cell_threshold = 120, cell_type_classifier = False)

    save_counts(save_structure, cell_coords)
    save_mips(save_structure)


def make_composite(mip_directory, save_directory):
    fileset = []
    for files in os.listdir(mip_directory):

        if files.endswith('tif'):
            fileset.append(os.path.join(mip_directory, files))

    fileset = fileset[::-1]
    images = []
    for tif in fileset:
        images.append(tifffile.imread(tif))

    images = np.asarray(images)

    finalfilename = path + '_' + item +'.tif'

    finalfilename = os.path.join(save_directory, finalfilename)
    tifffile.imsave(finalfilename,images)

    return finalfilename


def save_metadata(save_structure):
    save_path = os.path.join(save_structure["destination"], "metadata.txt")
    metadata = (
                "Last run on: " + str(date.today()) + "\n" + "\n" +
                "Red model version: " + RED_MODEL_NAME + "\n" +
                "Red probability threshold: " + str(RED_PROBABILITY_CUTOFF) + "\n" +
                "Red small object threshold: " + str(RED_SMALL_OBJECT_CUTOFF) + "\n" +
                "\n" +
                "Green model version: " + GREEN_MODEL_NAME + "\n" +
                "Green probability threshold: " + str(GREEN_PROBABILITY_CUTOFF) + "\n" +
                "Green small object threshold: " + str(GREEN_SMALL_OBJECT_CUTOFF) + "\n"
               )

    with open(save_path,'w') as f:
        f.write(metadata)

"""
Main
"""

cwd = os.getcwd()

basicxml = [os.path.join(cwd,f) for f in os.listdir(cwd) if 'basicxml' in f][0] #skeleton file for imagej plugin

root = Tk()
root.withdraw()
animal_id = filedialog.askdirectory(initialdir=cwd,title='Select the animal ID folder.')

red_model_path = os.path.join(RED_MODEL_NAME)
green_model_path = os.path.join(GREEN_MODEL_NAME)
red_model = init_model(red_model_path)
green_model = init_model(green_model_path)

destination_folder = os.path.join(animal_id, "cc_auto_results")
if not os.path.exists(destination_folder):
    os.mkdir(destination_folder)


for path in os.listdir(animal_id):

    save_structure = {"cwd" : cwd,
                      "animal" : animal_id,
                      "models" : (red_model, green_model),
                      "xml" : basicxml,
                      "destination" : destination_folder,
                      "save" : None,
                      "image" : None,
                      "composite" : None,
                      "coords" : None
                      }

    if "cc_auto_results" in path:
        continue

    datasetpath = os.path.join(animal_id, path)

    for item in os.listdir(datasetpath):

        subdir = os.path.join(datasetpath, item, 'MIP')
        crop_dir = os.path.join(subdir, 'cc_save_data')
        save_structure["save"] = crop_dir

        composite_name = make_composite(subdir, save_structure["destination"])
        save_structure["composite"] = composite_name

        run(save_structure)

        print("Completed counting for : " + os.path.basename(composite_name))
        print("---------------------------------------")

save_metadata(save_structure)
print("Saved, exiting.")
