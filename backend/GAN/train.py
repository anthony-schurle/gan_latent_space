import torch
import torchvision

class Train:
    def train(discriminator, generator, loss_function, device, learning_rate = 0.0001, num_epochs = 100, batch_size = 32):
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        optimizer_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate)

        #Transform that essentially subtracts all data by 0.5 and divides by 0.5 creating a range of [-1, 1]
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
        )

        #Get the training set of numbers of 28x28 pixels
        train_set = torchvision.datasets.MNIST(
            root=".", train=True, download=True, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        print("Training...")
        for epoch in range(num_epochs):
            for batch_num, (real_samples, mnist_labels) in enumerate(train_loader):
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
                if batch_num == batch_size - 1:
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")