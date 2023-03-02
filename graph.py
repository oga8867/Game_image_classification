import matplotlib.pyplot as plt
import numpy as np

plt.subplot(1, 2, 1)

plt.title("Training and Validation Loss")
plt.plot(np.arange(1, epoch + 1), val_losses, label="val")
plt.plot(np.arange(1, epoch + 1), train_losses, label="train")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
#    plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy")
plt.plot(np.arange(1, epoch + 1), val_accs, label="val")
plt.plot(np.arange(1, epoch + 1), train_accs, label="train")
plt.xlabel("epochs")
plt.ylabel("acuuracy")
plt.legend()

# plt.show()
plt.savefig("myfig_vit.png")
# fig.show()