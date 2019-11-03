from os import listdir, makedirs
from random import seed, random
from shutil import copyfile

from matplotlib import pyplot as plt
from sys import argv
from models import vgg_3block, get_iterators


def main():
    history = train()
    save_diagnostics(history)


def train():
    model = vgg_3block()
    train_it, test_it = get_iterators()
    history = model.fit_generator(train_it,steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=1)
    accuracy = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    return history


def save_diagnostics(hist):
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(hist.history['loss'], color='blue', label='train')
    plt.plot(hist.history['val_loss'], color='orange', label='test')

    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(hist.history['acc'], color='blue', label='train')
    plt.plot(hist.history['val_acc'], color='orange', label='test')

    filename = argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


def create_directories():
    rootdir = 'dataset_dogs_vs_cats/'
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        labeldirs = ['cats/', 'dogs/']
        for labeldir in labeldirs:
            newdir = rootdir + subdir + labeldir
            makedirs(newdir)


def transfer_photos_to_directories(sr=0.25):
    seed(1)
    split_rate = sr
    rootdir = 'dataset_dogs_vs_cats/'
    src_dir = 'train/'
    for file in listdir(rootdir + src_dir):
        src = rootdir + 'train/' + file
        dst_dir = 'train/'
        if random() < split_rate:
            dst_dir = 'test'
        animal_dir = 'dogs/'
        if file.startswith('cat'):
            animal_dir = 'cats/'
        dst = rootdir + dst_dir + animal_dir + file
        copyfile(src, dst)


if __name__ == '__main__':
    main()
