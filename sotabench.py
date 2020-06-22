import os

import torch
import torchvision.transforms as transforms
from torchbench.datasets.utils import download_file_from_google_drive
from torchbench.image_classification import ImageNet

from model.san import san
from util import config

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Model 1
config_path = 'config/imagenet/imagenet_san10_pairwise.yaml'
file_id = '1lv5TYfJFYvNWt_Ik0E-nAuI5h4PqSwuk'
destination = './tmp/'
filename = 'model_best.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
input_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

args = config.load_cfg_from_cfg_file(config_path)
model = san(args.sa_type, args.layers, args.kernels, args.classes)
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='SAN10-pairwise',
    paper_arxiv_id='2004.13621',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.749, 'Top 5 Accuracy': 0.921},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 2
config_path = 'config/imagenet/imagenet_san10_patchwise.yaml'
file_id = '1aU60a3I-YZK1HYs25sj2V5nbXC9FqRZ5'
destination = './tmp/'
filename = 'model_best.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
input_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

args = config.load_cfg_from_cfg_file(config_path)
model = san(args.sa_type, args.layers, args.kernels, args.classes)
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='SAN10-patchwise',
    paper_arxiv_id='2004.13621',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.771, 'Top 5 Accuracy': 0.935},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 3
config_path = 'config/imagenet/imagenet_san15_pairwise.yaml'
file_id = '1yfJnq28XYAjJ4ThYd9kv9MFBKFvzCr4I'
destination = './tmp/'
filename = 'model_best.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
input_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

args = config.load_cfg_from_cfg_file(config_path)
model = san(args.sa_type, args.layers, args.kernels, args.classes)
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='SAN15-pairwise',
    paper_arxiv_id='2004.13621',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.766, 'Top 5 Accuracy': 0.931},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 4
config_path = 'config/imagenet/imagenet_san15_patchwise.yaml'
file_id = '1MJwkzyo2wxjSCynj-Jp8NQMSj6o89Ph-'
destination = './tmp/'
filename = 'model_best.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
input_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

args = config.load_cfg_from_cfg_file(config_path)
model = san(args.sa_type, args.layers, args.kernels, args.classes)
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='SAN15-patchwise',
    paper_arxiv_id='2004.13621',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.780, 'Top 5 Accuracy': 0.939},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 5
config_path = 'config/imagenet/imagenet_san19_pairwise.yaml'
file_id = '1wAaGSizflxOCSiVHh2EE3mtyGL0IuPK9'
destination = './tmp/'
filename = 'model_best.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
input_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

args = config.load_cfg_from_cfg_file(config_path)
model = san(args.sa_type, args.layers, args.kernels, args.classes)
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='SAN19-pairwise',
    paper_arxiv_id='2004.13621',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.769, 'Top 5 Accuracy': 0.934},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()

# Model 6
config_path = 'config/imagenet/imagenet_san19_patchwise.yaml'
file_id = '1rtYIB-fTkDc4HR5s8K9Ddn2Xae-lhsfb'
destination = './tmp/'
filename = 'model_best.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
checkpoint = torch.load(os.path.join(destination, filename))
sd = {}
for key in checkpoint['state_dict']:
    sd[key.replace('module.', '')] = checkpoint['state_dict'][key]
# Define the transforms need to convert ImageNet data to expected model input
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
input_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

args = config.load_cfg_from_cfg_file(config_path)
model = san(args.sa_type, args.layers, args.kernels, args.classes)
model.load_state_dict(sd)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='SAN19-patchwise',
    paper_arxiv_id='2004.13621',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.782, 'Top 5 Accuracy': 0.939},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()
