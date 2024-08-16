import numpy as np
import cv2

class Gaussian:
    def __init__(self, mean, variance, weight):
        self.mean = mean
        self.variance = variance
        self.weight = weight

def initialize_gaussians(image, num_gaussians):
    height, width, _ = image.shape
    gaussians = np.zeros((height, width, num_gaussians), dtype=object)
    for i in range(height):
        for j in range(width):
            for k in range(num_gaussians):
                gaussians[i, j, k] = Gaussian(image[i, j], 15, 1 / num_gaussians)
    return gaussians

def adjust_factor(gaussians, frame, learning_rate=0.02):
    height, width, _ = frame.shape
    for i in range(height):
        for j in range(width):
            pixel = frame[i, j]
            for gaussian in gaussians[i, j]:
                if np.linalg.norm(pixel - gaussian.mean) < np.sqrt(gaussian.variance):
                    # Matching gaussian found, update parameters
                    rho = learning_rate * gaussian.weight
                    gaussian.mean = (1 - rho) * gaussian.mean + rho * pixel
                    gaussian.variance = (1 - rho) * gaussian.variance + rho * np.linalg.norm(pixel - gaussian.mean) ** 2
                    gaussian.weight += learning_rate * (1 - gaussian.weight)
                else:
                    # Not matching, reduce weight
                    gaussian.weight *= (1 - learning_rate)
            # Normalize the weights
            total_weight = sum(g.weight for g in gaussians[i, j])
            for gaussian in gaussians[i, j]:
                gaussian.weight /= total_weight

def update_gmm(frame, gaussians):
    height, width, _ = frame.shape
    foreground = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            pixel = frame[i, j]
            matches = [g for g in gaussians[i, j] if np.linalg.norm(pixel - g.mean) < np.sqrt(g.variance)]
            if matches:
                # Assign pixel to background
                foreground[i, j] = 0
            else:
                # Mark as foreground
                foreground[i, j] = 255
    return foreground

def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    num_gaussians = 3
    gaussians = initialize_gaussians(frame, num_gaussians)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        adjust_factor(gaussians, frame)
        foreground = update_gmm(frame, gaussians)
        
        cv2.imshow('Foreground', foreground)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
