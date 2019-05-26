import convolution
import numpy
import itertools

def get_args():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('image_path',help="Path to Image")
    argparser.add_argument('size',help="Convolution size, int",type=int)
    argparser.add_argument('--multiply',help="Color Multiplication (default: 1.0)",type=float,default=1.)
    args = argparser.parse_args()
    return args

# Read the image and get the numpy matrix associated with it. The return array is in the shape:
# (rows,cols,colors). There are 3 colors
def read_image(image_path):
    import matplotlib.pyplot as plt
    image = plt.imread(image_path)
    return image

# Saves the image into the file @param(name)
def save_image(image,name="newimg.png",cmap=False):
    import matplotlib.pyplot as plt
    if(not cmap):
        plt.imshow(image)
    else:
        plt.imshow(image,cmap='pink')
        plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight',pad_inches=0, transparent=True)
    print("image saved to {}".format(name))
    plt.clf()
    # plt.show()

if __name__ == "__main__":
    args = get_args()
    image = read_image(args.image_path)
    imagename = args.image_path[:args.image_path.find('.')]

    print('Applying convolution to image...')
    convolver = convolution.Convolver(image,power=2.)
    newimg = convolver.convolve(size=args.size)
    #newimg = convolution.multiply_color(newimg,args.multiply)

    avgimg = convolver.pool()
    diffimg = convolver.difference(newimg,avgimg)

    save_image(newimg,name="{}_conv.png".format(imagename))
    save_image(avgimg,name="{}_avg.png".format(imagename))
    save_image(diffimg,name="{}_diff_{}.png".format(imagename,args.size))
