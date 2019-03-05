from utils import scale, real_mse_loss, fake_mse_loss, cycle_consistency_loss, save_samples
import torch

def get_scaled_test(test_dataloader_x, test_dataloader_y):
    test_iter_x, test_iter_y = iter(test_dataloader_x), iter(test_dataloader_y)

    return (scale(test_iter_x.next()[0]),
            scale(test_iter_y.next()[0]))


def discriminator_step(discriminator, optimizer, forward_generator, input_images, output_images):
    optimizer.zero_grad()

    # Compute losses on real images
    real_loss = real_mse_loss( discriminator(input_images) )

    # Generate fake image
    fake_images = forward_generator(output_images)

    # Compute losses on fake images
    fake_loss = fake_mse_loss( discriminator(fake_images) )

    # Total loss and training step
    loss = real_loss + fake_loss
    loss.backward()
    optimizer.step()

    return loss


def generator_loss(straight_generator, forward_generator, discriminator, final_images):
    fake_input_images = forward_generator(final_images)
    forward_loss = real_mse_loss( discriminator(fake_input_images) )
    reconstructed_images = straight_generator(fake_input_images)
    reconstructed_loss = cycle_consistency_loss(final_images, reconstructed_images, lambda_weight=10)

    return forward_loss + reconstructed_loss


def generator_step(g_x_to_y, g_y_to_x, d_x, d_y, optimizer, images_x, images_y):
    optimizer.zero_grad()

    y_to_x_loss = generator_loss(g_x_to_y, g_y_to_x, d_x, images_y)
    x_to_y_loss = generator_loss(g_y_to_x, g_x_to_y, d_y, images_x)

    g_total_loss = y_to_x_loss + x_to_y_loss
    g_total_loss.backward()
    optimizer.step()

    return g_total_loss

def training_step(g_x_to_y, g_y_to_x, d_x, d_y, g_optimizer, d_x_optimizer, d_y_optimizer, iter_x, iter_y):
    images_x, _ = iter_x.next()
    images_x = scale(images_x)

    images_y, _ = iter_y.next()
    images_y = scale(images_y)

    if torch.cuda.is_available():
        images_x = images_x.to("cuda")
        images_y = images_y.to("cuda")

    d_x_loss = discriminator_step(d_x, d_x_optimizer, g_y_to_x, images_x, images_y)
    d_y_loss = discriminator_step(d_y, d_y_optimizer, g_x_to_y, images_y, images_x)
    g_loss = generator_step(g_x_to_y, g_y_to_x, d_x, d_y, g_optimizer, images_x, images_y)

    return (d_x_loss, d_y_loss, g_loss)


def training_loop(
        g_x_to_y, g_y_to_x,
        d_x, d_y, g_optimizer, d_x_optimizer, d_y_optimizer,
        dataloader_x, dataloader_y,
        test_dataloader_x, test_dataloader_y,
        n_epochs=1000):

    print_every = 10
    sample_every = 100

    losses = []

    fixed_x, fixed_y = get_scaled_test(test_dataloader_x, test_dataloader_y)

    iter_x, iter_y = iter(dataloader_x), iter(dataloader_y)

    batches_per_epoch = min(len(iter_x), len(iter_y))

    for epoch in range(1, n_epochs + 1):
        if epoch % batches_per_epoch == 0:
            iter_x, iter_y = iter(dataloader_x), iter(dataloader_y)

        d_x_loss, d_y_loss, g_loss = training_step(g_x_to_y, g_y_to_x, d_x, d_y, g_optimizer, d_x_optimizer, d_y_optimizer, iter_x, iter_y)

        if epoch % print_every == 0:
            losses.append((d_x_loss.item(), d_y_loss.item(), g_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_loss.item()))

        if epoch % sample_every == 0:
            g_y_to_x.eval()
            g_x_to_y.eval()
            save_samples(epoch, fixed_y, fixed_x, g_y_to_x, g_x_to_y, batch_size = 16)
            g_y_to_x.train()
            g_x_to_y.train()
