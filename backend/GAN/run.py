from . import generator as gen, discriminator as dis, device as dev, train

from PIL import Image
import torch
from torch import nn
#from pytorch_fid.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt
import os

def run():
    device = dev.Device().get_device()
    generator = gen.Generator().to(device=device)
    discriminator = dis.Discriminator().to(device=device)
    batch_size = 32
    train.Train.train(discriminator, generator, nn.BCELoss(), device, batch_size = batch_size)

    generate_srng(generator, batch_size, device)
    generate_srng(generator, batch_size, device)

def generate_srng(generator, batch_size, device):
    os.makedirs("out", exist_ok=True)

    latent_space_samples = torch.randn(batch_size, 100).to(device=device)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()

    for i in range(batch_size):
        img_tensor = generated_samples[i].reshape(28, 28)
        img_array = ((img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) * 255).numpy().astype('uint8')
        img = Image.fromarray(img_array, mode='L')
        img.save(os.path.join("out", f"srng_{i}.png"))

def generate_trng(generator, batch_size, device):
    os.makedirs("out", exist_ok=True)

    """
    #Replace with better random latent space
    with open("./gaussian.txt", "r") as f:
        numbers = [float(line.strip()) for line in f]
    numbers_tensor = torch.tensor(numbers, dtype=torch.float32, device=device)
    numbers_tensor = numbers_tensor[:batch_size * 100]
    latent_space_samples = numbers_tensor.view(batch_size, 100)
    generated_samples = generator(latent_space_samples)

    generated_samples = generated_samples.cpu().detach()
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
    plt.show()
    """
