#!/usr/bin/env python3
import numpy as np
import numpy.linalg as linalg
import os
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback

from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer
from scipy.spatial.transform import Rotation as R
from sklearn.pipeline import Pipeline

# adjust print options
np.set_printoptions(suppress=True, precision=3, linewidth=100, edgeitems=12)

# VR specific defines
input_vr_devices = 3
input_vr_frames = 1  # 3
output_vr_devices = 7
pretrained = False
preloaded = False

# Production, or quick iteration?
test_training = True

# https://stackoverflow.com/questions/53683164/
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()

# TODO: make this framerate OR framenum based, allow arbitrary frame shapes,
# e.g. [-1., -1, 0] for 'one second ago, one frame ago, current frame'
def augment_with_prior_frames(data_in, frame_shape=None, framerate=None):
    # TODO: use mocap.frame_time
    # TODO: normalize over-time head transform where head(t=0) = 0,0,z,
    # head(t=-1) = -dx_t, -dy_t, z
    back_1_s = data_in[0 : data_in.shape[0] - 60, :, :, :]
    back_1_frame = data_in[1 : data_in.shape[0] - 59, :, :, :]
    cur_frame = data_in[60 : data_in.shape[0],:, : ,:]
    return np.concatenate((back_1_s, back_1_frame, cur_frame), axis=1)

def load_frames_from_bvh(bvh_file):
    # load file
    parser = BVHParser()
    parsed_data = parser.parse(bvh_file)
    # get frames into matrix
    mp = MocapParameterizer('euler')
    # rotations now contains x,y,z + xrot,yrot,zrot for all channels
    rotations = mp.fit_transform([parsed_data])[0]
    # TODO: ALL data transforms (converting to t_m's, loading from bvh,
    # augmenting with prior frames) should be done as data pipe operations
    # for clarity
    data_pipe = Pipeline([
            ('param', MocapParameterizer('position')),
            #('rcpn', RootCentricPositionNormalizer()),
    ])

    positions = data_pipe.fit_transform([parsed_data])[0]
    head_xz_inverse_matrices = None
    dataset = np.zeros((len(rotations.values['Head_Xrotation']), 0, 3, 4))

    for joint_name in ['Head', 'LeftWrist', 'RightWrist',
                       'Hips',
                       'LeftAnkle', 'RightAnkle',
                       'LeftElbow', 'RightElbow',
                       'LeftKnee', 'RightKnee']:
        rot_frames = np.vstack(
            (rotations.values[joint_name + '_Xrotation'],
             rotations.values[joint_name + '_Yrotation'],
             rotations.values[joint_name + '_Zrotation']))
        r = R.from_euler('xyz', rot_frames.transpose(), degrees=True)
        pos_frames = np.vstack(
            (positions.values[joint_name + '_Xposition'],
             positions.values[joint_name + '_Yposition'],
             positions.values[joint_name + '_Zposition'])).transpose()[:, :, None]

        t_matrices = np.concatenate(
            (
                np.concatenate(
                    (
                        r.as_matrix(),
                        pos_frames,
                    ),
                    axis=2
                ),
                np.repeat(np.array([0., 0., 0., 1.])[None, None, :], dataset.shape[0], axis=0)
            ),
            axis=1
        )
        print("r: {} pos_frames: {} t: {}".format(r.as_matrix().shape,
                                                  pos_frames.shape,
                                                  t_matrices.shape))

        # Create an inverse-transform matrix for head's XZ position and
        # Y rotation, so everything's always head-relative except on the
        # vertical axis
        # TODO: do this pure-numpy and after data is loaded so we can
        # skip loading
        if joint_name == "Head":
            head_xz_rot_frames = np.vstack(
                (np.zeros((dataset.shape[0])),
                 -rotations.values[joint_name + '_Yrotation'],
                 np.zeros((dataset.shape[0]))))
            head_xz_r = R.from_euler('xyz', head_xz_rot_frames.transpose(), degrees=True)
            head_xz_pos_frames = np.vstack(
                (-positions.values[joint_name + '_Xposition'],
                 np.zeros((dataset.shape[0])),
                 -positions.values[joint_name + '_Zposition'])).transpose()[:, :, None]
            head_xz_inverse_matrices = np.concatenate(
                (
                    np.concatenate(
                        (
                            head_xz_r.as_matrix(),
                            head_xz_pos_frames,
                        ),
                        axis=2
                    ),
                    np.repeat(np.array([0., 0., 0., 1.])[None, None, :], dataset.shape[0], axis=0)
                ),
                axis=1
            )

        # Apply the inverse matrices to every transform (including the head)
        t_matrices = t_matrices @ head_xz_inverse_matrices

        # t_matrices now container an array of transform matrices, one for every
        dataset = np.concatenate((dataset,
                                  t_matrices[:, None, :3, :]),
                                 axis=1)
        print(dataset)

    # Normalize position scale [-1,1]
    pos_scale = np.max([np.max(comp) - np.min(comp)
                        for comp in dataset.transpose()])
    print("Scale factor: {} Overall shape: {}".format(pos_scale, dataset.shape))
    dataset *= np.array([[[1, 1, 1, 1./pos_scale],
                         [1, 1, 1, 1./pos_scale],
                         [1, 1, 1, 1./pos_scale]]]*10)
    return dataset


def load_all_bvh():
    if preloaded:
        with np.load('train_data_x.npz') as fi:
            x = fi[fi.files[0]]
        with np.load('train_data_y.npz') as fi:
            y = fi[fi.files[0]]
        return x, y

    accum_x = np.zeros((0, input_vr_devices * input_vr_frames, 3, 4))
    accum_y = np.zeros((0, output_vr_devices, 3, 4))
    for path in Path('input_bvh').rglob("*.bvh"):
        arr = load_frames_from_bvh(path.absolute())
        #if arr.shape[0] < 60:
            #continue
        #portion_x = augment_with_prior_frames(arr[:, :3, :, :])
        #portion_y = arr[:, 3:, :, :][60:]
        portion_x = arr[:, :input_vr_devices, :, :]
        portion_y = arr[:, input_vr_devices:, :, :]

        accum_x = np.append(accum_x, portion_x, axis=0)
        accum_y = np.append(accum_y, portion_y, axis=0)
        print("Accum shape, x: {} y: {}".format(accum_x.shape, accum_y.shape))
        cur_size = accum_x.size * accum_x.itemsize + accum_y.size * accum_y.itemsize
        print("Current size: {}B ({:.1f}M)".format(cur_size, cur_size/1024/1024))
        if cur_size > 256 * 1024 * 1024:
            print("Current size exceeded 2GB, exiting early")
            break


    # TODO: save raw, un-normalized loaded data as a different .npz so we can
    # iterate over it faster
    np.savez('train_data_x.npz', accum_x)
    np.savez('train_data_y.npz', accum_y)

    return accum_x, accum_y


# Our input set should be converted to shape (input_vr_devices*frames, 3, 4) and the
# output (output_vr_devices, 3, 4)
dataset_x, dataset_y = load_all_bvh()

# Split into train and validation sets using indicesing to optimize memory.
indices = np.arange(dataset_x.shape[0])
np.random.shuffle(indices)
train_indices = indices[: int(0.9 * dataset_x.shape[0])]
val_indices = indices[int(0.9 * dataset_x.shape[0]) :]

# Apply the processing function to the datasets - Discard the first 60 Y values, to align with X
x_train, y_train = dataset_x[train_indices], dataset_y[train_indices]
x_val, y_val = dataset_x[val_indices], dataset_y[val_indices]

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

# TODO: Some way to visualize some example data here

if pretrained:
    model = tf.keras.models.load_model('keras_model.h5')
else:
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(input_vr_devices * input_vr_frames, 3, 4))

    x = layers.Dropout(0.5)(inp)
    x = layers.Reshape((-1, 12))(x)
    x = layers.Permute((2, 1))(x)
    x = layers.LSTM(units=max(output_vr_devices, input_vr_devices * input_vr_frames), return_sequences=True, activation="tanh",)(x)
    x = layers.LSTM(units=max(output_vr_devices, input_vr_devices * input_vr_frames), return_sequences=True, activation="tanh",)(x)
    x = layers.LSTM(units=max(output_vr_devices, input_vr_devices * input_vr_frames), return_sequences=True, activation="tanh",)(x)
    x = layers.Dense(output_vr_devices, activation=None)(x)
    x = layers.Permute((2, 1))(x)
    x = layers.Reshape((-1, 3, 4))(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(
        run_eagerly=True,
        #loss=keras.losses.binary_crossentropy,
        loss=keras.losses.mse,
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
    epochs = 10 if test_training else 100
    batch_size = 5

    # Fit the model to the training data.
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr, ClearMemory()],
    )
    model.save('keras_model.h5', include_optimizer=False)
    try:
        os.remove('fdeep_model.json')
    except:
        pass

# Select a random example from the validation dataset.
example = x_val[np.random.choice(range(len(x_val)), size=1)[0]][:, :]

# Predict a new set of 10.
new_prediction = model.predict(np.expand_dims(example, axis=0))
print(repr(example[:,6:]))
print(repr(new_prediction))
