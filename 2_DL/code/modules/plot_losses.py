import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses):

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, 'b-', label = 'Training Loss')
    plt.plot(epochs, val_losses, 'r-', label = 'Validation Loss')

    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()