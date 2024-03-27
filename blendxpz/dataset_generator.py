import tensorflow as tf
import tensorflow_datasets as tfds
from madness_deblender import loadCATSIMDataset

def batched_CATSIMDataset_pz(
    tf_dataset_dir,
    linear_norm_coeff,
    batch_size,
    x_col_name="blended_gal_stamps",
    y_col_name="isolated_gal_stamps",
    train_data_dir=None,
    val_data_dir=None,
):
    """Load generated tf dataset.

    Parameters
    ----------
    tf_dataset_dir: string
        Path to generated tf dataset.
    linear_norm_coeff: int
        linear norm coefficient.
    batch_size: int
        size of batches to be generated.
    x_col_name: string
        column name for input to the ML models.
        Defaults to "blended_gal_stamps".
    y_col_name: string
        column name for loss computation of the ML models.
        Defaults to "isolated_gal_stamps".
    train_data_dir: string
        Path to .npy files for training.
        Ignored if TF Dataset is already present in output_dir.
    val_data_dir: string
        Path to .npy files for validation.
        Ignored if TF Dataset is already present in output_dir.

    Returns
    -------
    ds_train: tf dataset
        prefetched training dataset
    ds_val: tf dataset
        prefetched validation dataset

    """
    # normalized train and val dataset generator
    def preprocess_batch(ds):
        """Preprocessing function.

        Randomly flips, normalizes, shuffles the dataset

        Parameters
        ----------
        ds: tf dataset
            prefetched tf dataset

        Returns
        -------
        ds: tf dataset
            processing dataset, with (x,y) for training/validating network

        """

        def pre_process(elem):
            """Pre-processing function preparing data for denoising task.

            Parameters
            ----------
            elem: dict
                element of tf dataset.

            Returns
            -------
            (x, y): tuple
                data for training Neural Networks

            """
            x = elem[x_col_name] / linear_norm_coeff
            y = elem["pz"]

            do_flip_lr = tf.random.uniform([]) > 0.5
            if do_flip_lr:
                x = tf.image.flip_left_right(x)

            do_flip_ud = tf.random.uniform([]) > 0.5
            if do_flip_ud:
                x = tf.image.flip_up_down(x)

            return (x, y)

        ds = ds.shuffle(buffer_size=15 * batch_size)
        ds = ds.map(pre_process)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    ds = loadCATSIMDataset(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        output_dir=tf_dataset_dir,
    )

    ds_train = preprocess_batch(ds=ds[tfds.Split.TRAIN])

    ds_val = preprocess_batch(ds=ds[tfds.Split.VALIDATION])

    return ds_train, ds_val
