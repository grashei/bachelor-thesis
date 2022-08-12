from sklearn.model_selection import train_test_split
import os, glob

IMG_DIR = 'data/images'
MASK_DIR = 'data/masks'

images = [x for x in glob.glob(IMG_DIR + "/*.jpg")]
masks = [x for x in glob.glob(MASK_DIR+ "/*.png")]

images.sort()
masks.sort()


X_train, xtest, y_train, ytest = train_test_split(
    images, masks, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    xtest, ytest, test_size=0.5, random_state=42)

print(len(X_train), len(X_val), len(X_test))
print(len(y_train), len(y_val), len(y_test))

if not os.path.exists('training_ds'):
    os.makedirs('training_ds')

with open('training_ds/seg_train.txt', 'w+') as f:
    for x, y in zip(X_train, y_train):
        f.writelines(f'{x},{y}\n')

with open('training_ds/seg_val.txt', 'w+') as f:
    for x, y in zip(X_val, y_val):
        f.writelines(f'{x},{y}\n')

with open('training_ds/seg_test.txt', 'w+') as f:
    for x, y in zip(X_test, y_test):
        f.writelines(f'{x},{y}\n')
