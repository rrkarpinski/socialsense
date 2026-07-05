# socialsense

The project uses two datasets: MANNERSDB+ and MANNERSDB_OFFICE, both are extensions of the preexisting MANNERSDB.
Directory structure for the datasets is:
datasets
    MANNERSDBPlus
        NAO
        Pepper
        PR2
    OFFICE-MANNERSDB
        NAO
        Pepper
        PR2
Every robots dir structure:
[robot]
    Annotations
        *.csv - single CSV file with 11 columns: [IMAGE_ID, Vaccum Cleaning, Mopping the Floor, Carry Warm Food, Carry Cold Food, Carry Drinks, Carry Small Objects, Carry Large Objects, Cleaning, Starting a conversation, Reason]   
    Images
        [*.png] - multiple .png 1920x1080 images with names corresponding to IMAGE_ID's from the Annotations .csv file.


models/heuristicSplitModel_preprocessing.ipynb

models/heuristicSplitModel.py

buffers
baselines
training_utils.py


data_processing/


data_processing.ipynb

building dataset from raw data using buid_data.py 
prepare specific training datasets using data_utils.py


dataset analysis in data_analysis.ipynb

peturbations.ipynb to create occluded environmnet images. Other perturbations were done by manually editing images.

exeriments/
training
evaluation
statistical analysis of results using corrstats.py
resizing images for thesis document, various file comparisons, sanity checks - benchworking document

