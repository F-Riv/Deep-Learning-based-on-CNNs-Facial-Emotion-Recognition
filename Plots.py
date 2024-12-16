import matplotlib.pyplot as plt


def plot_history(history):
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()


def plot_histories(histories, labels):
    
    plt.figure(figsize=(12, 10))

    # Plot training & validation accuracy values
    plt.subplot(2, 2, 1)
    for history, label in zip(histories, labels):
        plt.plot(history['accuracy'], label=f'{label}')
      
    plt.title('Model Train Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(2, 2, 2)
    for history, label in zip(histories, labels):
        plt.plot(history['loss'], label=f'{label}')
     
    plt.title('Model Train Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')



     # Plot training & validation loss values
    plt.subplot(2, 2, 3)
    for history, label in zip(histories, labels):
   
        plt.plot(history['val_accuracy'], label=f'{label}')
    plt.title('Model Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')


     # Plot training & validation loss values
    plt.subplot(2, 2, 4)
    for history, label in zip(histories, labels):
        
        plt.plot(history['val_loss'], label=f'{label}')
    plt.title('Model Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()



def plot_history_from_file(history):
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()

    
        
def plot_instances_per_class(labels, all_labels):        
    counts = [sum(y == i for y in all_labels) for i in range(len(labels))]

    # Plot the number of images for each label
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel('Labels')
    plt.ylabel('Number of Images')
    plt.title('Number of Images in Training Dataset for Each Label')
    plt.xticks(rotation=45)
    plt.show()