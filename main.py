from model import create_model
from training_loop import training_loop
from utils import get_data_loader, display_dataset
from torch.optim import Adam

dataloader_X, test_dataloader_X = get_data_loader(image_type='summer')
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='winter')

display_dataset(dataloader_X)
display_dataset(dataloader_Y)

G_XtoY, G_YtoX, D_X, D_Y = create_model()

lr = 0.0002
beta1 = 0.5
beta2 = 0.999

g_params = list(G_XtoY.parameters()) + list(G_XtoY.parameters())

g_optimizer = Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = Adam(D_Y.parameters(), lr, [beta1, beta2])

n_epochs = 4000

training_loop(
        G_XtoY, G_YtoX,
        D_X, D_Y, g_optimizer, d_x_optimizer, d_y_optimizer,
        dataloader_X, dataloader_Y,
        test_dataloader_X, test_dataloader_Y,
        n_epochs)
