# Automatic Cell Counter

This package is meant to assist in the counting of starter cells for the mesoscale connectivity project. It consists of an ImageJ macro to format images,
and a python script which uses trained [DeepFLaSH](https://github.com/matjesg/DeepFLaSH) models in order to segment cells, and their count their overlap across channels. These overlapping cells are then saved in
an XML which is formatted to load into the ImageJ [cell counting plugin](https://imagej.nih.gov/ij/plugins/cell-counter.html).


### First Time Setup

1. Using an anaconda terminal, find the cell_count directory. Create an environment with the necessary libraries using `conda create --name cell_count --file spec_file.txt` and then `conda activate cell_count`.

2. Place **crop_mips.ijm** into your ImageJ -> Macros folder. Then, install via ImageJ when needed. Optionally, an [ImageJ hotkey](https://imagej.nih.gov/ij/developer/macro/macros.html) can be set up to allow quicker access.  

3. Access the ML models via [Sharepoint](https://alleninstitute.sharepoint.com/sites/PWA/Imaging/Shared%20Documents/Forms/AllItems.aspx?). Drag the **red_model.h5** and **green_model.h5** files into your cell_count directory.


### Executing on Full Datasets

1. Enter the cell_count Anaconda environment by running `conda activate cell_count` in your terminal.

2. Copy the desired animal ID folder anywhere on your desktop.

3. Run crop_mips.ijm in ImageJ, and select the copied animal ID folder.

4. Using the rectangle pointer tool, adjust the given ROI rectangle to include the cells in the MIP image.

5. Using the terminal, run **cell_count.py**. Select the copied animal ID folder again.

6. Open the newly created cc_auto_results folder.

7. Initialize the composite MIP images using the cell counter plugin. Load the corresponding XML coordinate files.

8. Confirm cell placements, and adjust thresholds/parameters as necessary. It may be helpful to look in cc_save_data (inside MIP folder) to analyze the segmented images.


### Executing on Single MIP Images

1. Enter the cell_count Anaconda environment by running `conda activate cell_count` in your terminal.

2. Copy the desired MIP folder anywhere on your desktop.

3. Run crop_mips.ijm in ImageJ, and select the copied MIP folder.

4. Using the rectangle pointer tool, adjust the given ROI rectangle to include the cells in the MIP image.

5. Using the terminal, run **cell_count.py**. Select the newly created **cc_save_data** folder.

6. Inside of **cc_save_data**, a new composite image and corresponding cell count coordinates should be made.


### Tips and Adjustments

#### Cell Type Marking System

In the beginning stages of this package, a marking system is in place in order to allow users to understand the behavior of the cell counter. When cell_type_classifier is set to True, the cell types in the ImageJ cell counter plugin will be used to denote different attributes of cells that were read by the counter. Currently, cell type 1 denotes positive counts, cell type 2 denotes positive counts that were suspiciously large, and cell type 3 denotes negative counts that overlapped, but not through the centroid. These cell type markers can be modified to display most any classification desired to understand the behavior of the cell counter, and can also be turned off to facilitate proper counting.

#### Thresholding and Adjustable Arguments

There are a small number of manually determined variables that affect the performance of the counter. It is worth the user's time to learn how these adjustments affect the segmented images, and modify them as needed for optimal performance. For each iteration of **cell_count.py** that runs, a metatdata.txt file is saved with the settings used for that count.

- **probability_cutoff:** The prediction returned by the DeepFLaSH model gives each pixel a value between 0 and 1, denoting the model's confidence that the pixel in question belongs to a cell. Probability_cutoff determines the minimum confidence required when translating this probability image into a binary one. In current practice, the green channel requires a lower probability_cutoff than the red channel does.

- **small_object_cutoff:** Before labelling each object in the image, all objects which contain fewer pixels than this argument denotes will be removed. Again, the green channel tends to require a lower cutoff than the red channel does.


#### Best Practices

- Sometimes, the ImageJ cropping rectangle will not appear. This is okay, as the rectangle is just a manual reference, and the program will crop in a 1024 x 1024 rectangle no matter what. Simply draw a new rectangle which is roughly 1000 x 1000, and place it normally before pressing OK.

- Try to avoid cropping down the middle of cells of interest. The model doesn't tend to like this very much, and may miss these during segmentation.

- Try to include as little noise as possible. If there are areas of intense red than can be cropped without losing any cells, it will lead to better performance. The models have been trained to ignore noise, but don't do a perfect job.

- Look out for extremely populated areas, Especially in the red channel. The current trained models may undercount when the divisions between cells are not clear.


### Training a new model

If the trained models aren't up to the task, they can always be trained further. This requires 5-10 image-mask pairs, for each model to be fitted. Once obtained, a new model can be trained using the DeepFLaSH [Google Collab](https://colab.research.google.com/github/matjesg/DeepFLaSH/blob/master/DeepFLaSH.ipynb) page. Full instructions for this page can be found [here](https://github.com/matjesg/DeepFLaSH/raw/master/user_guide.pdf).


#### Google Collab

If training from an included model, the model must be loaded into Collab, and the script must be directed toward it. Do this by creating a copy of the Collab on your google drive. Drag the model to be trained into the Files tab on the left of the page. Then, insert the following lines into the "model choice" code block, as shown:

```
Model_name = "cFOS_Wue"
model_path = os.path.join("MODEL_NAME_HERE.h5")
model = utils.load_model(model_path, custom_objects={"recall": utils.recall,
                                            "precision": utils.precision,
                                            "f1": utils.f1,
                                            "mcor": utils.mcor,
                                            "weighted_bce_dice_loss": utils.weighted_bce_dice_loss})

```
Upload your images and masks, named according to the Collab procedure. Train for 50-100 epochs, and do some preliminary tests on the new model. When satisfied, download the new model using the files tab. Make sure to store all old models, and rename the models in use accordingly.


#### Creating Image Masks

1. Start with either a labelled image from running count_cells, or with a probability mask returned from running the Collab on your model. Open in ImageJ.
2. Threshold the image. Maintain values close to the ones provided by the **probability_cutoff** variables for the channel being trained.
3. Make sure your ROI Manager is empty. Then, under Edit, find Selection -> Create Selection. Press ctrl + t to add the selection to the ROI Manager.
4. Using the ROI manager, select the added ROI and then find More -> Split. Delete the original ROI, which should be at the top of the ROI Manager. Sometimes, the last ROI on the list may also need to be deleted.
5. Open the cropped MIP image which the mask belongs to, and click Show All under the ROI Manager.
6. Add ROIs using the Freehand Selection tool, and delete ROIs as needed. When complete, the ROI list should represent a ground-truth cell segmentation.
7. When ready, under the ROI Manager, click Deselect, then More -> OR (Combine).
8. Under Edit, click Selection -> Create Mask.
9. If the Image title bar says Inverted LUT, invert the Lookup Tables under Image -> Lookup Tables.
10. Save the mask, named according to the practices set by the Collab.
