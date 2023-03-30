import cv2
import numpy as np

def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold(image: np.ndarray, method: str = "otsu", thresh: int = 127) -> np.ndarray:
    if method == "otsu":
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return binary_image

def denoise(image: np.ndarray, method: str = "median", kernel_denoise: int = 3) -> np.ndarray:
    if method == "median":
        return cv2.medianBlur(image, kernel_denoise)
    else:
        return cv2.GaussianBlur(image, (kernel_denoise, kernel_denoise), 0)

def sharpen(image: np.ndarray, amount: float = 2, kernel_size: int = 3) -> np.ndarray:
    blurred_image = cv2.GaussianBlur(image, (0, 0), kernel_size)
    sharpened_image = cv2.addWeighted(image, 1 + amount, blurred_image, -amount, 0)
    return sharpened_image

def generate_grid(num_rows, num_cols, image_size=(1038, 2162)):
    grid = np.zeros((num_rows * image_size[0], num_cols * image_size[1], 3))
    for i in range(num_rows):
        for j in range(num_cols):
            img = cv2.imread('outputs/binary_image_{}.png'.format((1+i * num_cols + j)))
            grid[i * image_size[0]: (i + 1) * image_size[0], j * image_size[1]: (j + 1) * image_size[1]] = img
    cv2.imwrite('./binary_grid_.png', grid)

def main(kernel_size: int, kernel_denoise: int, amount: int):
    image = cv2.imread("example.png")
    if image is None:
        print("Error: Could not open the image file.")
        return

    gray_image = grayscale(image)
    denoised_image = denoise(gray_image, 'gauss', kernel_denoise=kernel_denoise)
    sharpened_image = sharpen(denoised_image, amount, kernel_size)
    binary_image = threshold(sharpened_image)
    

    # save the image
    # cv2.imwrite('outputs/gray_image.png', gray_image)
    # cv2.imwrite('outputs/denoised_image.png', denoised_image)
    # cv2.imwrite('outputs/sharpened_image.png', sharpened_image)
    cv2.imwrite('outputs/binary_image_{}.png'.format(kernel_size), binary_image)

for i in range(1, 21, 2):
    main(i, 5, 5)

generate_grid(4, 5, image_size=(1038, 2162))