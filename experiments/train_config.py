import torch
from data_processing.data_utils import get_transform, IMAGENET_NORM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
domains = ['Home', 'BigOffice-2', 'BigOffice-3', 'Hallway', 'MeetingRoom', 'SmallOffice']

testing_scenarios = {
    # 'normal.onlysoc': ('onlysoc', 120, "../data/mean_data_pepper_train.pkl", "../data/mean_data_pepper_test.pkl", None, [get_transform((144,256), IMAGENET_NORM)], ['image_path_soc'], False),
    'hdf5.onlysoc': ('onlysoc', 120, None, None, "../data/mean_data_pepper.hdf5", [get_transform((144,256), IMAGENET_NORM)], ['image_path_soc'], False),
}
