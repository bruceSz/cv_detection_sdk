#!/usr/bin/env python

from framework.train_config import CIFARConfig as Config

def main():
    
    
    conf = Config()
    train_loader, val_loader = get_loaders(conf)
    train_loop(conf, VGG16_CF(), train_loader, val_loader, 'vgg16_cf')


if __name__ == "__main__":
    main()