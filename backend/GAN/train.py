import torch

class Train:
    def train(discriminator, generator, loss_function, device, learning_rate, num_epochs, batch_size, train_set):
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True
        )

        print("Training...")
        for epoch in range(num_epochs):
            for batch_num, (real_samples, labels) in enumerate(train_loader):
                # Data for training the discriminator
                real_samples = real_samples.to(device=device)
                real_samples_labels = torch.ones((batch_size, 1)).to(
                    device=device
                )
                latent_space_samples = torch.randn((batch_size, 100)).to(
                    device=device
                )
                generated_samples = generator(latent_space_samples)
                generated_samples_labels = torch.zeros((batch_size, 1)).to(
                    device=device
                )
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Training the discriminator
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples)
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels
                )
                loss_discriminator.backward()
                optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((batch_size, 100)).to(
                    device=device
                )

                # Training the generator
                generator.zero_grad()
                generated_samples = generator(latent_space_samples)
                output_discriminator_generated = discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                optimizer_generator.step()

            # Show loss
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")