import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from pymo.parsers import BVHParser
import io
from pymo.viz_tools import draw_stickfigure, print_skel
from pymo.preprocessing import MocapParameterizer, RootCentricPositionNormalizer, JointSelector, Numpyfier
from scipy.spatial.transform import Rotation as R
from sklearn.pipeline import Pipeline

# adjust print options
np.set_printoptions(suppress = True, precision=3, linewidth=100, edgeitems=12)

# VR specific defines
input_vr_devices = 3
input_vr_frames = 3
output_vr_devices = 7
pretrained = False
preloaded = False

# Production, or quick iteration?
test_training = True

# TODO: make this framerate OR framenum based, allow arbitrary frame shapes,
# e.g. [-1., -1, 0] for 'one second ago, one frame ago, current frame'
def augment_with_prior_frames(data_in, frame_shape=None, framerate=None):
    # TODO: use mocap.frame_time
    back_1_s = data_in[0 : data_in.shape[0] - 60, :, :]
    back_1_frame = data_in[1 : data_in.shape[0] - 59, :, :]
    cur_frame = data_in[60 : data_in.shape[0],:, : ]
    return np.concatenate((back_1_s, back_1_frame, cur_frame), axis=2)

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

    dataset = np.zeros((len(rotations.values['Head_Xrotation']), 12, 0))
    head_starting_positions = None
    for joint_name in ['Head', 'LeftWrist', 'RightWrist',
                       'Hips',
                       'LeftAnkle', 'RightAnkle',
                       'LeftElbow', 'RightElbow',
                       'LeftKnee', 'RightKnee']:
        rot_frames = np.vstack(
            (rotations.values[joint_name + '_Xrotation'],
             rotations.values[joint_name + '_Yrotation'],
             rotations.values[joint_name + '_Zrotation'])).transpose()
        r = R.from_euler('xyz', rot_frames, degrees=True)
        pos_frames = np.vstack(
            (positions.values[joint_name + '_Xposition'],
             positions.values[joint_name + '_Yposition'],
             positions.values[joint_name + '_Zposition'])).transpose()[:,None]

        # Note, in 3ds Max convention, Z should be up - but it's not clear yet
        # what convention this data uses.
        # TODO: Normalize positions relative to head
        #if joint_name == "Head":
            #head_starting_positions = pos_frames
        #pos_frames -= head_starting_positions

        print(r.as_matrix().shape)
        print(pos_frames.shape)
        t_matrices = np.concatenate(
            (r.as_matrix(),
             np.swapaxes(pos_frames, 1, 2)),
            axis=2
        )
        # print(t_matrices)
        # t_matrices now container an array of transform matrices, one for every
        dataset = np.dstack((dataset,t_matrices.reshape(-1,12)))

    # TODO: normalize position to where all non-head transforms are
    # relative to the head transform

    # Normalize position scale [-1,1]
    pos_scale = np.max([np.max(comp) - np.min(comp)
                        for comp in dataset.transpose()])
    print(pos_scale)
    dataset *= np.array([[1,1,1,1./pos_scale,
                         1,1,1,1./pos_scale,
                         1,1,1,1./pos_scale]]*10).transpose()
    print(dataset.shape)
    print(dataset)
    #print_skel(parsed_data)
    #draw_stickfigure(positions[0], frame=0)
    return dataset


def load_all_bvh():
    if preloaded:
        with np.load('train_data_x.npz') as fi:
            x = fi[fi.files[0]]
        with np.load('train_data_y.npz') as fi:
            y = fi[fi.files[0]]
        return x, y

    accum_x = np.zeros((0, 12, 9))
    accum_y = np.zeros((0, 12, 7))
    for path in Path('input_bvh').rglob("*.bvh"):
        arr = load_frames_from_bvh(path.absolute())
        if arr.shape[0] < 60:
            continue
        portion_x = augment_with_prior_frames(arr[:,:,:3])
        portion_y = arr[:,:,3:][60:]

        print(portion_x.shape)
        print(portion_y.shape)
        accum_x = np.append(accum_x, portion_x, axis=0)
        accum_y = np.append(accum_y, portion_y, axis=0)

    # TODO: normalize over-time head transform where head(t=0) = 0,0,z,
    # head(t=-1) = -dx_t, -dy_t, z
    print(accum_x.shape)
    print(accum_y.shape)
    np.savez('train_data_x.npz', accum_x)
    np.savez('train_data_y.npz', accum_y)

    return accum_x, accum_y


# Our input set should be converted to shape (12, input_vr_devices*frames) and the
# output (12, output_vr_devices)
dataset_x, dataset_y = load_all_bvh()

# TODO: translation of non-head transforms should be normalized relative to
# the head transform - need a way to avoid the X and Y of the head from changing
# too much

# Split into train and validation sets using indicesing to optimize memory.
indices = np.arange(dataset_x.shape[0])
np.random.shuffle(indices)
train_indices = indices[: int(0.9 * dataset_x.shape[0])]
val_indices = indices[int(0.9 * dataset_x.shape[0]) :]

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

if pretrained:
    model = tf.keras.models.load_model('keras_model.h5')
else:
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(12,input_vr_devices * input_vr_frames)) #shape=(None, *x_train.shape[2:]))

    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    #x = layers.LSTM(units=4, return_sequences=True, activation="relu",)(inp)
    #x = layers.LSTM(units=8, return_sequences=True, activation="relu",)(x)
    #x = layers.LSTM(units=12, return_sequences=True, activation="relu",)(x)
    #x = layers.Dropout(0.2)(x)
    #x = layers.Dense(output_vr_devices, activation="sigmoid")(x)
    x = layers.Dropout(0.4)(inp)
    x = layers.LSTM(units=9, return_sequences=True, activation="tanh",)(x)
    x = layers.LSTM(units=9, return_sequences=True, activation="tanh",)(x)
    x = layers.LSTM(units=9, return_sequences=True, activation="tanh",)(x)
    x = layers.Dense(output_vr_devices, activation="tanh")(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(
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
        callbacks=[early_stopping, reduce_lr],
    )
    model.save('keras_model.h5', include_optimizer=False)

# Select a random example from the validation dataset.
example = x_val[np.random.choice(range(len(x_val)), size=1)[0]][:, :]

# Predict a new set of 10.
new_prediction = model.predict(np.expand_dims(example, axis=0))
print(repr(example[:,6:]))
print(repr(new_prediction))
