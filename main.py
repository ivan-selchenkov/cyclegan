from utils import get_data_loader, display_dataset

dataloader_X, test_dataloader_X = get_data_loader(image_type='summer')
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='winter')

display_dataset(dataloader_X)
display_dataset(dataloader_Y)