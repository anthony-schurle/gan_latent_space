from . import generator as gen, discriminator as dis, device as dev, train

from PIL import Image
import torch
from torch import nn
import torchvision
from pytorch_fid import fid_score
import os

def run(batch_size: int = 32):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = torchvision.datasets.MNIST( root=".", train=True, download=True, transform=transform)

    label_to_idx = dict()
    for idx, (_, label) in enumerate(train_set):
        label = label
        if label not in label_to_idx.keys():
            label_to_idx[label] = []
        label_to_idx[label].append(idx)

    device = dev.Device().get_device()

    os.makedirs("out/fid", exist_ok=True)
    srng_fid_seq0 = []
    srng_fid_seq1 = []
    srng_fid_seq2 = []
    trng_fid_seq = []
    for label in range(10):
        print(f"GAN Label - {label}:\n")
        train_subset = torch.utils.data.Subset(train_set, label_to_idx[label])
        generator = gen.Generator().to(device=device)
        discriminator = dis.Discriminator().to(device=device)

        train.Train.train(discriminator, generator, nn.BCELoss(), device, 0.0001, 500, batch_size, train_subset)

        srng_path0 = f"out/srng0/label-{label}"
        srng_path1 = f"out/srng1/label-{label}"
        srng_path2 = f"out/srng2/label-{label}"
        trng_path = f"out/trng/label-{label}"
        train_path = f"out/training/label-{label}"
        generate_srng(generator, 300, device, srng_path0, 67)
        generate_srng(generator, 300, device, srng_path1, 69)
        generate_srng(generator, 300, device, srng_path2, 222)
        generate_trng(generator, 300, device, trng_path)
        show_training_data(train_subset, f"out/training/label-{label}")

        print("Calculating FID...")
        srng_fid_seq0.insert(0, fid_score.calculate_fid_given_paths([train_path, srng_path0], batch_size, device=device.type, dims=2048))
        srng_fid_seq1.insert(0, fid_score.calculate_fid_given_paths([train_path, srng_path1], batch_size, device=device.type, dims=2048))
        srng_fid_seq2.insert(0, fid_score.calculate_fid_given_paths([train_path, srng_path2], batch_size, device=device.type, dims=2048))
        trng_fid_seq.insert(0, fid_score.calculate_fid_given_paths([train_path, trng_path], batch_size, device=device.type, dims=2048))

    with open("out/fid/fid.txt", "w") as file:
        for label in range(len(srng_fid_seq0)):
            file.write(f"Label - {label}:\n")
            file.write(f"\tSRNG - {srng_fid_seq0[label]}\n")
            file.write(f"\tSRNG - {srng_fid_seq1[label]}\n")
            file.write(f"\tSRNG - {srng_fid_seq2[label]}\n")
            file.write(f"\tTRNG - {trng_fid_seq[label]}\n")
        file.write("Average:\n")
        file.write(f"SRNG - {sum(srng_fid_seq0) / len(srng_fid_seq0)}\n")
        file.write(f"SRNG - {sum(srng_fid_seq1) / len(srng_fid_seq1)}\n")
        file.write(f"SRNG - {sum(srng_fid_seq2) / len(srng_fid_seq2)}\n")
        file.write(f"TRNG - {sum(trng_fid_seq) / len(trng_fid_seq)}\n")

def generate_srng(generator, gen_size, device, output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed)
    latent_space_samples = torch.randn(gen_size, 100).to(device=device)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()

    for i in range(gen_size):
        img_tensor = generated_samples[i].reshape(28, 28)
        img_array = ((img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) * 255).numpy().astype('uint8')
        img = Image.fromarray(img_array, mode='L').convert("RGB")
        img.save(os.path.join(output_dir, f"srng_{i}.png"))
    
    return generated_samples

def generate_trng(generator, gen_size, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open("./gaussian.txt", "r") as f:
        numbers = [float(line.strip()) for line in f]
    numbers_tensor = torch.tensor(numbers, dtype=torch.float32, device=device)
    numbers_tensor = numbers_tensor[:gen_size * 100]
    latent_space_samples = numbers_tensor.view(gen_size, 100)

    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()

    for i in range(gen_size):
        img_tensor = generated_samples[i].reshape(28, 28)
        img_array = ((img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) * 255).numpy().astype('uint8')
        img = Image.fromarray(img_array, mode='L').convert("RGB")
        img.save(os.path.join(output_dir, f"trng_{i}.png"))
    
    return generated_samples

def show_training_data(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(data)):
        img_tensor, label = data[i]
        img_tensor = img_tensor.squeeze(0)
        img_array = ((img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) * 255).numpy().astype('uint8')
        img = Image.fromarray(img_array, mode='L').convert("RGB")
        img.save(os.path.join(output_dir, f"training_{i}.png"))
