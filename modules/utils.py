import sys
import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_samples(conv_net, params, data_handler):
    samples = defaultdict(list)
    accuracies = {}
    for dataset_key in ['train', 'test']:
        n_batches = data_handler.num_batches(dataset_key)
        num_correct = 0
        total_num = 0
        for batch_idx, (images, targets) in enumerate(data_handler(dataset_key)):
            # Stdout
            sys.stdout.write(f'\rEvaluating {dataset_key} set | Batch {batch_idx+1}/{n_batches}...')
            sys.stdout.flush()
            # Compute target class
            target_class = np.argmax(targets, axis=1)

            # Model Inference
            predicted_class = np.argmax(conv_net(params, images), axis=1)
            
            num_correct += np.sum(predicted_class==target_class)
            total_num += images.shape[0]

            samples[dataset_key].append((images, target_class, predicted_class))
        
        acc = num_correct / total_num
        accuracies[dataset_key] = acc
        print(f"\nAccuracy: {acc:0.2f}")
    return samples, accuracies

def plot_metrics(train_loss, accuracies):
    fig, ax = plt.subplots(1,2, figsize=(18,4))

    # Training loss subplot
    train_loss_flat = [ele2 for ele in train_loss for ele2 in ele]
    ax[0].plot(train_loss_flat)
    ax[0].set(title='Training Loss vs. # Training Steps', ylabel='Training Loss', xlabel='# Training Steps')
    
    # Accuracies subplot
    for key, val in accuracies.items():
        epochs = onp.array(val)[:,0]
        acc_vals = onp.array(val)[:,1]
        ax[1].plot(epochs, acc_vals, label=key)

    ax[1].set(title='Accuracy vs. # Epochs', ylabel='Accuracy', xlabel='# Epochs')
    ax[1].legend()

def show_predictions(samples, accuracies, data_handler, n_vis_samples=4):
    renorm = lambda im : (im - im.min()) / (im.max() - im.min())
    
    for key, vals in samples.items():
        accuracy = accuracies[key]
        fig, ax = plt.subplots(1, n_vis_samples, figsize=(12,12))
        ax[0].text(52, -10, f"{key.capitalize()} Set Accuracy: {accuracy:0.2f}", fontsize=18, style='oblique')
        
        idxs = onp.random.choice(range(len(vals)), n_vis_samples, replace=False)
        for i, idx in enumerate(idxs):
            images, target_class, predicted_class = vals[idx]
            im = images[0].squeeze()
            if len(im.shape) == 1:
                im = onp.reshape(im, data_handler.img_dim[1:])
            
            if len(im.shape) == 3:
                im = onp.transpose(im, (1,2,0))
            im = renorm(im)
            cmap = None if len(im.shape)==3 else 'gray'

            yi = target_class[0]
            y_label = data_handler.classes[yi].capitalize()

            pi = predicted_class[0]
            p_label = data_handler.classes[pi].capitalize()

            title = f'Actual: {y_label}\nPrediction: {p_label}'
            ax[i].imshow(im, cmap=cmap)
            ax[i].set_title(title)
            ax[i].axis('off')
        plt.show()
        plt.clf()

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