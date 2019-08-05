import os
from tkinter import filedialog, Tk
from collections import OrderedDict
import datetime
import ast

import xmltodict as xtd
import tifffile as tf
import numpy as np
import scipy.ndimage as ndi

from keras.models import load_model
from unet.metrics import recall, precision, f1, mcor
from unet.losses import weighted_bce_dice_loss
from unet import utils

from skimage.morphology import remove_small_objects, watershed
from skimage.measure import regionprops, label
from skimage.transform import resize
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.util import img_as_int

RED_PROBABILITY_CUTOFF = .55
RED_SMALL_OBJECT_CUTOFF = 20

GREEN_PROBABILITY_CUTOFF = .3
GREEN_SMALL_OBJECT_CUTOFF = 10

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


class Counter:

    def __init__(self, name, models, crop_dir, destination_dir = None):

        self.name = name

        self.cwd = os.getcwd()
        self.models = models
        self.cc_save_data = crop_dir

        if destination_dir is None:
            self.cc_auto_results = self.cc_save_data

        else:
            self.cc_auto_results = destination_dir

        self.basicxml = [os.path.join(self.cwd, f) for f in os.listdir(self.cwd) if 'basicxml' in f][0] #skeleton file for imagej plugin
        self.mip_dir = sorted([os.path.join(self.cc_save_data,f) for f in os.listdir(self.cc_save_data) if (f.endswith('tif') and ("label" not in f))])
        self.crop_coord_dir = [os.path.join(self.cc_save_data,f) for f in os.listdir(self.cc_save_data) if (f.endswith('coords.txt'))][0]


    def execute(self):
        """Runs the cell counting procedure for a Counter object.

        Loads and segments the MIP images using load_images() and segment(). Counts the cells and saves
        the coordinates and segmented images using count_overlap(), save_labelled(), and save_counts(). Assumes the existence
        of a cc_save_data folder containing ch01 and ch02 cropped images, as well as crop coordinates in a .txt file.

        """

        mip_images = self.load_images()

        r_labelled, r_props = self.segment(mip_images[1], self.models[0], probability_cutoff = RED_PROBABILITY_CUTOFF, small_object_cutoff = RED_SMALL_OBJECT_CUTOFF)
        g_labelled, g_props = self.segment(mip_images[0], self.models[1], probability_cutoff = GREEN_PROBABILITY_CUTOFF, small_object_cutoff = GREEN_SMALL_OBJECT_CUTOFF)

        self.labelled = (r_labelled, g_labelled)
        self.props = (r_props, g_props)

        self.cell_coords = self.count_overlap(double_cell_threshold = 120, cell_type_classifier = False)

        self.save_counts()
        self.save_composite()
        self.save_labelled()


    def load_images(self):
        """Finds the MIP pathway, and returns the images loaded in tifffile."""

        images = []
        for i in range(len(self.mip_dir)):

            full_img = tf.imread(self.mip_dir[i])
            img = resize(full_img, (1024, 1024, 1)) #converts from 4d np array to 3d
            images.append(img)

        return images


    def segment(self, img, model, probability_cutoff = .50, small_object_cutoff = 30):
        """Uses a DeepFLaSH model to segment a 1024 x 1024 image, then binarizes at probability_cutoff. Watersheds and labels the image.

        model.predict() returns an array with shape (1, 1024, 1024, 1) in which every pixel's value is the model's confidence (from 0 to 1)
        that a cell exists in that pixel. This image is thresholded at the user-determined probability_cutoff. A watershed is performed,
        and then objects smaller than small_object_cutoff are removed. Each remaining object in the binary image is labelled.

        Parameters
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
        image_predict = np.reshape(image_predict, (1024, 1024))

        cutoff_indexes = image_predict <= probability_cutoff
        image_predict[cutoff_indexes] = 0 #adjusts the image to only include pixels above probability_cutoff
        binary_indexes = image_predict > 0
        image_predict[binary_indexes] = 1 #sets all other pixels to 1

        image_predict = image_predict.astype(int)

        distance = ndi.distance_transform_edt(image_predict)
        distance_blur = gaussian(distance, sigma = 1.2, preserve_range = True)
        local_maxi = peak_local_max(distance_blur, indices = False, footprint = np.ones((8, 8)), labels = image_predict)
        markers = ndi.label(local_maxi)[0]
        image_watershed = watershed(-distance_blur, markers, mask = image_predict, watershed_line = True)

        image_labelled = label(image_watershed)
        image_cleaned = remove_small_objects(image_labelled, small_object_cutoff, connectivity = 1, in_place = False)
        image_labelled, num = label(image_cleaned, return_num = True)
        r_props = regionprops(image_labelled)

        print(str(num) + " labelled cells.")

        return image_labelled, r_props


    def count_overlap(self, double_cell_threshold = 85, cell_type_classifier = False):
        """Iterates over the red objects, and looks for overlapping green objects. Stores coordinates of overlap.

        Each red object's centroid location is accessed in the green labelled image. If the coordinates belong to an object in
        the green prop list, the green object's centroid location is accessed in the red image. If these coordinates refer to the
        original red object, the red centroid coordinates are appended to cell_coords.

        Optionally, various cell types can be marked by different properties. If cell_type_classifier is True, cell types
        other than "positive" will be recorded, and later stored as different cell markers in imageJ.

        Parameters
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

        red_props, green_props = self.props[0], self.props[1]
        red_labelled, green_labelled = self.labelled[0], self.labelled[1]

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


    def save_counts(self):
        """Writes all cell (x, y) coordinates into an xml file readable by the imageJ cellcount plugin.

        xmltodict is used to read and write xml files as dictionaries. All coordinates are offset by the amount
        recorded in self.crop_coord_dir, to account for the original cropping of the image.

        Parameters
        ----------
        cell_coords : dictionary of lists containing cell coordinates and classifications, as returned from count_overlap.

        Returns
        -------
        None

        """

        with open(self.basicxml,'r') as f:
            cell_count_xml = xtd.parse(f.read())  #turns xml skeleton into dictionary

        with open(self.crop_coord_dir) as f:
            offset = f.readline()
        offset = ast.literal_eval(offset) #reads the string as a tuple, for immediate typecasting.
        x_off, y_off = offset[0], offset[1]

        cell_counts = [[] for _ in range(len(self.cell_coords))]

        for cell_type in self.cell_coords.keys():

            if cell_type == "positive":
                cell_marker_type = 0
            elif cell_type == "double":
                cell_marker_type = 1
            else:
                cell_marker_type = 2

            if len(self.cell_coords[cell_type]) != 0:

                for cell in self.cell_coords[cell_type]:

                    x, y = cell[0], cell[1]

                    marker_dict = OrderedDict()
                    marker_dict['MarkerX'] = y + x_off #offset added to account for cropping
                    marker_dict['MarkerY'] = x + y_off  #coordinates in XML counter are flipped !?
                    marker_dict['MarkerZ'] = 1

                    cell_counts[cell_marker_type].append(marker_dict)

                cell_count_xml['CellCounter_Marker_File']["Marker_Data"]['Marker_Type'][cell_marker_type]["Marker"] = cell_counts[cell_marker_type]

                cell_count_xml['CellCounter_Marker_File']["Image_Properties"]["Image_Filename"] = self.name #if this name doesn't match the image name, coordinates will not load

        xml_id = "CellCounter_" + self.name.replace(".tif", ".xml")

        save_xml = os.path.join(self.cc_auto_results, xml_id)
        xml_string = xtd.unparse(cell_count_xml, pretty=True, newl="\n",indent="  ")

        with open(save_xml,'w') as f:
            f.write(xml_string)


    def save_labelled(self):
        """Saves the segmented and labelled red and green images to the crop subdirectory with type uint16."""

        red_filename = self.name.replace(".tif", "_red_label.tif")
        green_filename = self.name.replace(".tif", "_green_label.tif")

        save_red_label = os.path.join(self.cc_save_data, red_filename)
        save_green_label = os.path.join(self.cc_save_data, green_filename)

        red_label = self.labelled[0]
        green_label = self.labelled[1]

        tf.imsave(save_red_label, red_label.astype(np.uint16)) #type conversion for image accessibility in imageJ
        tf.imsave(save_green_label, green_label.astype(np.uint16))


    def save_composite(self):
        """Looks in mip_directory for .tif files, then combines them into a composite image."""

        mip_directory = os.path.dirname(self.cc_save_data)

        fileset = []
        for file in os.listdir(mip_directory):
            if file.endswith('tif'):
                fileset.append(os.path.join(mip_directory, file))

        fileset = fileset[::-1]
        images = []
        for tif in fileset:
            images.append(tf.imread(tif))

        images = np.asarray(images) #creates composite from list of single-channel images.

        finalfilename = os.path.join(self.cc_auto_results, self.name)
        tf.imsave(finalfilename, images)



def save_metadata(destination_folder):
    """Creates a metadata.txt file in the composite directory and writes all global variables to it, along with the current date."""

    save_path = os.path.join(destination_folder, "metadata.txt")
    red_time = os.path.getctime(os.path.join(os.getcwd(), RED_MODEL_NAME))
    green_time = os.path.getctime(os.path.join(os.getcwd(), GREEN_MODEL_NAME))

    metadata = (
                "Last run on: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" + "\n" +

                "Red model version: " + RED_MODEL_NAME + " (created " + datetime.datetime.fromtimestamp(red_time).strftime('%c') + ")\n" +
                "Red probability threshold: " + str(RED_PROBABILITY_CUTOFF) + "\n" +
                "Red small object threshold: " + str(RED_SMALL_OBJECT_CUTOFF) + "\n" +
                "\n" +
                "Green model version: " + GREEN_MODEL_NAME + " (created " + datetime.datetime.fromtimestamp(green_time).strftime('%c') + ")\n" +
                "Green probability threshold: " + str(GREEN_PROBABILITY_CUTOFF) + "\n" +
                "Green small object threshold: " + str(GREEN_SMALL_OBJECT_CUTOFF) + "\n"
               )

    with open(save_path,'w') as f:
        f.write(metadata)


def run_single(crop_dir, models):
    """Creates a Counter object using directory and model parameters. Executes."""

    counter = Counter("single_count.tif", models, crop_dir)
    counter.execute()

    print("Completed counting for : " + counter.name)
    print("-------------------------------------")

    save_metadata(crop_dir)
    print("Saved, exiting.")


def run_multiple(animal_id, models):
    """Parses through the animal_id directory to find all MIP folders, then creates a Counter object for each. Executes.

    Creates a cc_auto_results folder in which to store the count xml files and their composite MIP counterparts.

    Parameters
    ----------
    animal_id : string, directory of the brain folder.
    models : loaded DeepFLaSH model objects.

    Returns
    -------
    None

    """

    destination_folder = os.path.join(animal_id, "cc_auto_results")
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    for path in os.listdir(animal_id):
        if "cc" in path: #avoids scanning composites as they are created
            continue

        datasetpath = os.path.join(animal_id, path)

        for item in os.listdir(datasetpath):

            subdir = os.path.join(datasetpath, item, 'MIP')
            crop_dir = os.path.join(subdir, 'cc_save_data')

            composite_name = path + '_' + item + '.tif'

            counter = Counter(composite_name, models, crop_dir, destination_folder)
            counter.execute()

            print("Completed counting for : " + counter.name)
            print("-------------------------------------")

    save_metadata(destination_folder)
    print("Saved, exiting.")


def main():
    """Prompts the user to select a folder, then loads ML models and calls run() on every available MIP.

    Either run_single or run_multiple is called, depending on the folder type selected.
    """

    root = Tk()
    root.withdraw()
    chosen_folder = filedialog.askdirectory(initialdir = os.getcwd(), title = "Select the animal ID folder.")

    if chosen_folder == "":
        raise Exception("No folder was chosen.")

    red_model_path = os.path.join(RED_MODEL_NAME)
    green_model_path = os.path.join(GREEN_MODEL_NAME)

    red_model = init_model(red_model_path)
    green_model = init_model(green_model_path)

    models = (red_model, green_model)

    if "cc_save_data" in chosen_folder:
        print("Single MIP folder chosen.")
        run_single(chosen_folder, models)

    else:
        print("Animal folder chosen.")
        run_multiple(chosen_folder, models)


if __name__ == "__main__":
    main()
