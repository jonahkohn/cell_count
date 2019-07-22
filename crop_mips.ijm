
macro "Crop Mips" {

    chosen_folder = getDirectory("Select the animal ID or MIP folder");

    if (endsWith(chosen_folder, "MIP\\")){
        load_single(chosen_folder);
    }
    else{
        load_multiple(chosen_folder);
    }

} // macro


function crop(path, destination, x, y) {

    makeRectangle(x, y, 1024, 1024);
    run("Crop");
    saveAs("tif", destination);

} //crop


function load_multiple(animal_id){

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

                        execute(red_mip_path, "");

                    } //if
                } // for ch
            } // for j
        } // else
    } // for i
} //load_multiple


function load_single(mip_folder){

    mip_images = getFileList(mip_folder);

    for (ch = 0; ch < mip_images.length; ch++) {

        if (endsWith( mip_images[ch], "ch02.tif")){
            red_mip_path = mip_folder + "\\" + mip_images[ch];
            red_mip_path = replace(red_mip_path, "//", "/");
            red_mip_path = replace(red_mip_path, "/", "\\");

            execute(red_mip_path, "_single");

        } //if
    } // for ch
} //load_single


function execute(red_path, folder_extension) {

    open(red_path);
    close("\\Others");
    setBatchMode(true);

    red_filename = File.getName(red_path);
    directory = File.getParent(red_path);
    savepath = directory + "/cc_save_data" + folder_extension;
    File.makeDirectory(savepath);

    makeRectangle(0, 0, 1024, 1024);
    waitForUser("Move the rectangle over the cells of interest, then press OK.");
    getSelectionCoordinates(xPoints,yPoints);
        x = xPoints[0];
        y = yPoints[0];  //x and y refer to the top left corner of the rectangle.

    coords = "("+x+","+y+")";  //this format is directly readable as a python tuple.
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
