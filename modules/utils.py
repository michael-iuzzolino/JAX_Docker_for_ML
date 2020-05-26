import numpy as onp
import matplotlib.pyplot as plt

def plot_metrics(train_loss, accuracies):
    fig, ax = plt.subplots(1,2, figsize=(18,4))

    # Training loss subplot
    training_steps = onp.array(train_loss)[:,0]
    training_loss_vals = onp.array(train_loss)[:,1]
    ax[0].plot(training_steps, training_loss_vals)
    ax[0].set(title='Training Loss vs. # Training Steps', ylabel='Training Loss', xlabel='# Training Steps')
    # Accuracies subplo
    for key, val in accuracies.items():
        epochs = onp.array(val)[:,0]
        acc_vals = onp.array(val)[:,1]
        ax[1].plot(epochs, acc_vals, label=key)

    ax[1].set(title='Accuracy vs. # Epochs', ylabel='Accuracy', xlabel='# Epochs')
    ax[1].legend()
    
def show_samples(handler, n_vis_samples=4, dataset_key='train'):
    renorm = lambda im : (im - im.min()) / (im.max() - im.min())
    for x, y in handler(dataset_key):
        if x.shape[1:] != handler.img_dim:
            x = onp.reshape(x, (*x.shape[:1], *handler.img_dim))
        x = x.squeeze()
        break
    
    sample_idxs = onp.random.choice(range(x.shape[0]), n_vis_samples, replace=False)

    fig, ax = plt.subplots(1, n_vis_samples, figsize=(12,12))
    for i, sample_idx in enumerate(sample_idxs):
        im = x[sample_idx]
        if len(im.shape) == 3:
            im = onp.transpose(im, (1,2,0))
        im = renorm(im)
        cmap = None if len(im.shape)==3 else 'gray'
        yi = y[sample_idx]
        if isinstance(yi, onp.ndarray):
            yi = onp.argmax(yi)
        label = handler.classes[yi].capitalize()
        ax[i].imshow(im, cmap=cmap)
        ax[i].set_title(label)
        ax[i].axis('off')
        
def label_2_onehot(labels, num_classes):
    return onp.eye(num_classes)[labels]