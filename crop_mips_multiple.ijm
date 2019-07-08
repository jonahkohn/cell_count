


macro "Cell Count Crop [H]" {

    animal_id = getDirectory("Select the animal ID folder");
    datasets = getFileList(animal_id);

    for (i = 0; i < datasets.length; i++) {

        if (endsWith(datasets[i], "cc_auto_results/")) {
            continue;
        } // if

        else {
            datasets_path = animal_id + datasets[i];
            items = getFileList(datasets_path);

            for (j = 0; j < items.length; j++) {
                mip_folder_path = datasets_path + items[j] + "/MIP";
                mip_images = getFileList(mip_folder_path);

                for (ch = 0; ch < mip_images.length; ch++) {

                    if (endsWith( mip_images[ch], "ch02.tif")){
                        red_mip_path = mip_folder_path + "\\" + mip_images[ch];
                        red_mip_path = replace(red_mip_path, "//", "/");
                        red_mip_path = replace(red_mip_path, "/", "\\");

                        execute(red_mip_path);

                } // for ch
            } // for j
        } // else
    } // for i
} // macro


function crop(path, destination, x, y) {

    makeRectangle(x, y, 1024, 1024);
    run("Crop");
    saveAs("tif", destination);
}


function execute(red_path) {

    open(red_path);
    close("\\Others");
    setBatchMode(true);

    red_filename = File.getName(red_path);
    directory = File.getParent(red_path);
    savepath = directory + "/cc_save_data";
    File.makeDirectory(savepath);

    makeRectangle(0, 0, 1024, 1024);
    waitForUser("Move the rectangle over the cells of interest, then press OK.");
    getSelectionCoordinates(xPoints,yPoints);
        x = xPoints[0];
        y = yPoints[0];

    coords = "("+x+","+y+")";
    coord_file =  replace(red_filename, "ch02.tif", "coords.txt");
    File.saveString(coords, savepath + "/" + coord_file);

    red_crop_file = replace(red_filename, ".tif", "_crop.tif");
    red_destination = savepath + "/" + red_crop_file;

    crop(red_path, red_destination, x, y);

    green_path = replace(red_path, "ch02.tif", "ch01.tif");
    green_filename = File.getName(green_path);
    green_crop_file = replace(green_filename, ".tif", "_crop.tif");
    green_destination = savepath + "/" + green_crop_file;

    open(green_path);
    crop(green_path, green_destination, x, y);

    close(red_filename);
    close(green_filename);

    setBatchMode(false);

}
