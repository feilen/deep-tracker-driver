import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import io

# VR specific defines
input_vr_devices = 3
input_vr_frames = 3
output_vr_devices = 7
TMP_RAND_COUNT = 1000

def augment_with_prior_frames(data_in):
    back_1_s = data_in[0 : data_in.shape[0] - 60, :, :]
    back_1_frame = data_in[1 : data_in.shape[0] - 59, :, :]
    cur_frame = data_in[60 : data_in.shape[0],:, : ]
    print(str(cur_frame.shape))
    return np.concatenate((back_1_s, back_1_frame, cur_frame), axis=2)

# Our input set should be converted to shape (12, input_vr_devices) and the
# output (12, output_vr_devices)
dataset_x = augment_with_prior_frames(np.random.rand(TMP_RAND_COUNT,12,input_vr_devices))
dataset_y = np.random.rand(TMP_RAND_COUNT,12,output_vr_devices)[60:]

# TODO: translation of non-head transforms should be normalized relative to
# the head transform - need a way to avoid the X and Y of the head from changing
# too much

# Split into train and validation sets using indicesing to optimize memory.
indices = np.arange(dataset_x.shape[0])
np.random.shuffle(indices)
train_indices = indices[: int(0.9 * dataset_x.shape[0])]
val_indices = indices[int(0.9 * dataset_x.shape[0]) :]

# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
#def create_shifted_frames(data):
#    x = data[:, 0 : data.shape[1] - 1, :, :]
#    y = data[:, 1 : data.shape[1], :, :]
#    return x, y

# TODO In practice, we're actually going to want the most recent two frames,
# and a frame 1s ago - let's sample that by going 60 frames back

# Apply the processing function to the datasets - Discard the first 60 Y values, to align with X
# TODO: augment_with_... must happen BEFORE we shuffle the data
x_train, y_train = dataset_x[train_indices], dataset_y[train_indices]
x_val, y_val = dataset_x[val_indices], dataset_y[val_indices]

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

# TODO: Some way to visualize some example data here

# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(12,input_vr_devices * input_vr_frames)) #shape=(None, *x_train.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.LSTM(units=4, return_sequences=True, activation="relu",)(inp)
x = layers.LSTM(units=8, return_sequences=True, activation="relu",)(x)
x = layers.LSTM(units=12, return_sequences=True, activation="relu",)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(output_vr_devices, activation="sigmoid")(x)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(),
)
model.summary()

"""
## Model Training

With our model and data constructed, we can now train the model.
"""

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 20
batch_size = 5

# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
)
assert(False)
"""
## Frame Prediction Visualizations

With our model now constructed and trained, we can generate
some example frame predictions based on a new video.

We'll pick a random example from the validation set and
then choose the first ten frames from them. From there, we can
allow the model to predict 10 new frames, which we can compare
to the ground truth frame predictions.
"""

# Select a random example from the validation dataset.
example = val_dataset[np.random.choice(range(len(val_dataset)), size=1)[0]]

# Pick the first/last ten frames from the example.
frames = example[:10, ...]
original_frames = example[10:, ...]

# Predict a new set of 10 frames.
for _ in range(10):
    # Extract the model's prediction and post-process it.
    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_prediction = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame), axis=0)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Plot the new frames.
new_frames = frames[10:, ...]
for idx, ax in enumerate(axes[1]):
    ax.imshow(np.squeeze(new_frames[idx]), cmap="gray")
    ax.set_title(f"Frame {idx + 11}")
    ax.axis("off")

# Display the figure.
plt.show()

"""
## Predicted Videos

Finally, we'll pick a few examples from the validation set
and construct some GIFs with them to see the model's
predicted videos.

You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/conv-lstm)
and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/conv-lstm).
"""

# Select a few random examples from the dataset.
examples = val_dataset[np.random.choice(range(len(val_dataset)), size=5)]

# Iterate over the examples and predict the frames.
predicted_videos = []
for example in examples:
    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]
    new_predictions = np.zeros(shape=(10, *frames[0].shape))

    # Predict a new set of 10 frames.
    for i in range(10):
        # Extract the model's prediction and post-process it.
        frames = example[: 10 + i + 1, ...]
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

        # Extend the set of prediction frames.
        new_predictions[i] = predicted_frame

    # Create and save GIFs for each of the ground truth/prediction images.
    for frame_set in [original_frames, new_predictions]:
        # Construct a GIF from the selected video frames.
        current_frames = np.squeeze(frame_set)
        current_frames = current_frames[..., np.newaxis] * np.ones(3)
        current_frames = (current_frames * 255).astype(np.uint8)
        current_frames = list(current_frames)

        # Construct a GIF from the frames.
        with io.BytesIO() as gif:
            #imageio.mimsave(gif, current_frames, "GIF", fps=5)
            predicted_videos.append(gif.getvalue())

## Display the videos.
#print(" Truth\tPrediction")
#for i in range(0, len(predicted_videos), 2):
#    # Construct and display an `HBox` with the ground truth and prediction.
#    box = HBox(
#        [
#            widgets.Image(value=predicted_videos[i]),
#            widgets.Image(value=predicted_videos[i + 1]),
#        ]
#    )
#    display(box)
