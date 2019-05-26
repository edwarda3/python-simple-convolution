import numpy
from itertools import product
import tqdm

class Convolver:
    # normalize the image so that values are floats from 0-1. 
    # This allows proper behavior of the powersum() function.
    def __init__(self,image,**kwargs):
        self.power = kwargs.get('power',1.)
        if len(image.shape) <= 2:
            self.image = numpy.zeros((image.shape[0],image.shape[1],3))
            self.image[:,:,0] = numpy.copy(image)
            self.image[:,:,1] = numpy.copy(image)
            self.image[:,:,2] = numpy.copy(image)
        else:
            self.image = image
        self.image = normalize(self.image)

    # Raise all elements in the window to a power, then sum them.
    def powersum(self,window):
        return numpy.sum(numpy.power(window,self.power))

    # Perform convolution with window size @param(size). 
    # Return a slightly shrunk image, with the parts that do not have a full convolution window trimmed.
    def convolve(self,*,size=3):
        print("Performing convolution with window size {}".format(size))
        halfsize = size//2
        rem = size % 2
        v_bound = (0+halfsize, self.image.shape[0]-halfsize)
        h_bound = (0+halfsize, self.image.shape[1]-halfsize)
        
        convolved = numpy.zeros((self.image.shape[0],self.image.shape[1],3))

        #for (row,col,color) in tqdm.tqdm(product(range(*v_bound),range(*h_bound),range(3))):
        for row in tqdm.trange(*v_bound):
            for col in range(*h_bound):
                for color in range(self.image.shape[2]):
                    # Center the convolution window around the (row,col) pixel and take the box that is size=size. Add all elements in that window together after raising their power to self.power.
                    # Since this is a normalized value between 0-1, it can only get darker.
                    window = self.image[row-halfsize:row+halfsize+rem,col-halfsize:col+halfsize+rem,color].reshape(size*size)
                    convolved[row,col,color] = self.powersum(window)
        
        convolved = convolved / (size*size) # Take the average of the window sum
        return trim(convolved,halfsize)

    #Perform convolution multiple times, then take the average of the result. The size will be equal to the size of the largest convolution window (smallest returned image).
    def pool(self):
        # sizes = [5,11,21,41,61]
        sizes = [3,5,7,11,15,21]
        pooled = []
        for size in sizes:
            image = self.convolve(size=size)
            pooled.append(trim(image,(max(sizes)-size)//2))

        print('Finding the Average of all vectors')
        avgimg = numpy.zeros(pooled[0].shape)
        for row in tqdm.trange(pooled[0].shape[0]):
            for col in range(pooled[0].shape[1]):
                for color in range(pooled[0].shape[2]):
                    for i in range(len(pooled)):
                        avgimg[row,col,color] += (pooled[i][row,col,color]) * (1/len(pooled))
        return avgimg

    # Get the difference between two vectors. This is made to be used on an image with a single convolution and the average from pooled.
    def difference(self,single,avg):
        print('Finding the difference between two vectors')
        if(single.shape[0] < avg.shape[0]):
            print('The average image must be smaller than or equal to the single image.')
            return False
        sizediff = single.shape[0] - avg.shape[0]
        single_c = trim(single,sizediff//2)
        for row in tqdm.trange(single_c.shape[0]):
            for col in range(single_c.shape[1]):
                for color in range(single_c.shape[2]):
                    single_c[row,col,color] = abs(single_c[row,col,color] - avg[row,col,color])
        return single_c


# Normalizes the vector to be a float from 0-1
def normalize(image):
    newimg = numpy.copy(image)
    newimg = newimg / 255.
    return newimg
    
# Trims off @param(margin) pixels from each side
def trim(image,margin):
    trimmed = numpy.copy(image)
    return trimmed[margin:image.shape[0]-margin,margin:image.shape[1]-margin,:]

# Multiplies each color value by @param(factor). This results in a brightened or darkened image, depending on if @param(factor) > 1 or otherwise.
def multiply_color(image,factor):
    newimg = numpy.zeros(image.shape)
    for (row,col,color) in product(*map(range,image.shape)):
        newimg[row,col,color] = min(max(image[row,col,color] * factor, 0.),1.)
    return newimg