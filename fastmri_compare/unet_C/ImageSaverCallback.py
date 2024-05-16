import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

class ImageSaverCallback(Callback):
    def __init__(self, val_gen, save_path, freq=1):
        super().__init__()
        self.val_gen = val_gen
        self.save_path = save_path
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            # Get one batch of data from the validation generator
            sample_batch = next(iter(self.val_gen))
            input_image = sample_batch[0][0]
            target_image = sample_batch[1][0]

            # Predict the output
            predicted_image = self.model.predict(np.expand_dims(input_image, axis=0))[0]

            # Save images
            self.save_images(input_image, target_image, predicted_image, epoch)

    def save_images(self, input_image, target_image, predicted_image, epoch):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(input_image.squeeze(), cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(abs(target_image.squeeze()), cmap='gray')
        axes[1].set_title('Target Image')
        axes[1].axis('off')

        axes[2].imshow(predicted_image.squeeze(), cmap='gray')
        axes[2].set_title('Predicted Image')
        axes[2].axis('off')

        plt.suptitle(f'Epoch {epoch+1}')
        plt.savefig(f'{self.save_path}/epoch_{epoch+1}.png')
        plt.close(fig)
