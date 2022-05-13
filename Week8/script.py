import numpy as np
from PIL import Image

kernel_one = np.array([[-1,-1,-1],[-1,-8,-1],[-1,-1,-1]])
kernel_two = np.array([[0,-1,0],[-1,-8,-1],[0,-1,0]])

def apply_convolution_kernel(input, kernel):
    input_length, input_width = input.shape
    kernel_length, kernel_width = kernel.shape

    width_diff = input_width - kernel_width
    length_diff = input_length - kernel_length
    convolved_array = np.zeros((input_length-2, input_width-2))
     
    for row in range(length_diff+1):
        for col in range(width_diff+1):
            out = np.sum(input[row:row+kernel_width, col:col+kernel_length] * kernel)
            convolved_array[row][col] = out

    return convolved_array


im = Image.open ('twitter.jpg')
rgb = np.array(im.convert('RGB'))
r=rgb[:,:,0]
out = apply_convolution_kernel(r, kernel_one)
Image.fromarray(np.uint8(out)).show()

result = apply_convolution_kernel(r, kernel_two)
Image.fromarray(np.uint8(out)).show()

