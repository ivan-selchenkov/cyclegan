from utils import scale, real_mse_loss, fake_mse_loss
import torch

def get_scaled_test(test_dataloader_x, test_dataloader_y):
    test_iter_x, test_iter_y = iter(test_dataloader_x), iter(test_dataloader_y)

    return (scale(test_iter_x.next()[0]),
            scale(test_iter_y.next()[0]))


def discriminator_step(discriminator, optimizer, back_generator, input_images, output_images):
    optimizer.zero_grad()

    # Compute losses on real images
    real_loss = real_mse_loss( discriminator(input_images) )

    # Generate fake image
    fake_images = back_generator(output_images)

    # Compute losses on fake images
    fake_loss = fake_mse_loss( discriminator(fake_images) )

    # Total loss and training step
    loss = real_loss + fake_loss
    loss.backward()
    optimizer.step()


def generator_step(optimizer):
    optimizer.zero_grad()


def training_step(g_x_to_y, g_y_to_x, d_x, d_y, d_x_optimizer, d_y_optimizer, iter_x, iter_y):
    images_x, _ = iter_x.next()
    images_x = scale(images_x)

    images_y, _ = iter_y.next()
    images_y = scale(images_y)

    if torch.cuda.is_available():
        images_x = images_y.to("cuda")
        images_y = images_y.to("cuda")

    discriminator_step(d_x, d_x_optimizer, g_y_to_x, images_x, images_y)
    discriminator_step(d_y, d_y_optimizer, g_x_to_y, images_y, images_x)



def training_loop(g_x_to_y, g_y_to_x, d_x, d_y, d_x_optimizer, d_y_optimizer, dataloader_x, dataloader_y, test_dataloader_x, test_dataloader_y, n_epochs=1000):
    print_every = 10

    losses = []

    fixed_x, fixed_y = get_scaled_test(test_dataloader_x, test_dataloader_y)

    iter_x, iter_y = iter(dataloader_x), iter(dataloader_y)

    batches_per_epoch = min(len(iter_x), len(iter_y))

    for epoch in range(1, n_epochs + 1):
        if epoch % batches_per_epoch == 0:
            iter_x, iter_y = iter(dataloader_x), iter(dataloader_y)

        training_step(g_x_to_y, g_y_to_x, d_x, d_y, d_x_optimizer, d_y_optimizer, iter_x, iter_y)



