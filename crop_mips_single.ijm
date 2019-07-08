macro "Cell Count Crop [H]" {

    red_path = File.openDialog("Select the red MIP image");

    open(red_path);
    red_filename = File.getName(red_path);
    directory = File.getParent(red_path);
    savepath = directory + "/cc_save_data_single";
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

}

function crop(path, destination, x, y) {

    makeRectangle(x, y, 1024, 1024);
    run("Crop");
    saveAs("tif", destination);
}
