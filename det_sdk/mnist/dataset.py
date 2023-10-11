#!/usr/bin/env python3

import torch 
import torchvision
def create_loader(root="../data/mnist"):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root, train=True, download=False ,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,),(0.3081,)
                                        )
                                    ])),batch_size=64, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root, train=False, download=False,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307, ), (0.3081,)
                                        )
                                    ])), batch_size=64, shuffle=True)
    return train_loader, test_loader


def test_create_loader():
    root = "../data/mnist"
    train_loader, test_loader = create_loader(root)
    examples = enumerate(train_loader)
    batch_idx, (data, target) = next(examples)
    print(batch_idx)
    print("total train examples: ", len(train_loader))


if __name__ == '__main__':
    test_create_loader()