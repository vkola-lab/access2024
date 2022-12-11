import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from skimage import morphology


def rescale(array, tup) -> tuple:
    """
    rescale Feature scales image to range tup

    _extended_summary_

    Parameters
    ----------
    array : np.ndarray
        ndarray to feature scale
    tup : len(2) tuple, range to scale to
        range to which we feature scale the image

    Returns
    -------
    tuple
        (feature scaled array, (min, range, original range))
    """
    m = np.min(array)
    if m < 0:
        array += -m
    a = np.max(array) - np.min(array)
    t = tup[1] - tup[0]
    return (array * t / a), (m, t, a)


def rescale_from_prev(array, m, t, a) -> tuple:
    """
    rescale_from_prev _summary_

    _extended_summary_

    Parameters
    ----------
    array : _type_
        _description_
    m : min
        minimum to use for scaling (output of rescale)
    t : range
        range from original image (output of rescale)
    a : original range
        original range from output of rescale


    Returns
    -------
    tuple
        (feature scaled array, (min, range, original range))
    """
    if m < 0:
        array += -m
    return array * t / a, (m, t, a)


def read_txt(fname: str):
    with open(fname, "r") as fi:
        return eval(fi.readlines()[0])


class BrainSample(object):
    """
    BrainSample loads a brain and can plot the brain in axial or coronal dimensions, including
    either the original image, the slice removed images, vanilla gan, or new gan images. You can
    also overlay images, and plot in color or grayscale

    Parameters
    ----------
    brain_path: string
        base path to directory containing subdirectories as follows:
            T = original images
            Z = slice removed images
            G = vanilla GAN images
            CG_1 = new GAN images
    rid: string
        4 digit string specifying the subject

    """

    prefix = "masked_brain_mri_"
    file_type = ".nii"
    folder_map = {
        "T": "orig",
        "Z": "noised",
        "G": "vanilla",
        "CG_1": "novel",
        "M": "mask",
    }
    idx_rel_path = "slice_list/ADNI"

    def __init__(self, brain_path="/Users/mromano/research/data/rgan_data", rid="0307"):
        self.rid = rid
        self.path = brain_path
        self.output_path = os.path.join(brain_path, "..", "img")
        self.brains = {}
        for f, nm in self.folder_map.items():
            self.brains[nm] = NiftiBrain.from_file(
                brain_path, f, self.prefix + self.rid + self.file_type
            )
        self.idx_missing = read_txt(
            os.path.join(brain_path, self.idx_rel_path, self.prefix + self.rid + ".txt")
        )
        self.idx_missing.sort()

    def plot_slice(
        self,
        dim: str = "z",
        type_: str = "orig",
        num: str = "32",
        threshold=1e-06,
        caxis: tuple = (0, 2.5),
        colorbar: bool = False,
    ):
        """
        plot_slice Plots a single slice from one of the Brains

        Parameters
        ----------
        dim : str, optional
            can be x, y, or z depending on desired orientation, by default 'z'
        type_ : str, optional
            can be any of "noised", "orig", "vanilla", "novel", or "mask, by default 'orig'
        num : str, optional
            the number of the slice, by default '32'
        threshold : _type_, optional
            voxel value threshold, by default 1e-06
        caxis : tuple, optional
            color axis range, by default (0, 2.5)
        colorbar : bool, optional
            include colorbar?, by default False
        """
        os.makedirs(self.output_path, exist_ok=True)
        ax = plt.subplot(111)
        plotting.plot_anat(
            self.brains[type_].brain_img,
            axes=ax,
            display_mode=dim,
            cut_coords=[num],
            threshold=threshold,
            annotate=False,
            vmax=caxis[1],
            vmin=caxis[0],
            colorbar=colorbar,
            black_bg=False,
        )
        f_name = (
            f"{self.output_path}/{self.rid}_{dim}_"
            f"{str(num).zfill(3)}_{type_}_colorbar_{str(colorbar)}.eps"
        )
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.close()

    def plot_missing_montage(self):
        """
        plot_missing_montage _summary_

        _extended_summary_
        """
        slices = []
        for idx in np.arange(5, len(self.idx_missing) - 5, 10):
            curr_slices = []
            for val in ["orig", "noised", "vanilla", "novel"]:
                curr_slices.append(
                    np.squeeze(
                        self.brains[val].original_brain[self.idx_missing[idx], :, :]
                    )
                )
            curr_slices = np.concatenate(curr_slices, axis=1).T
            slices.append(curr_slices)
        slices = np.concatenate(slices, axis=1)
        ax = plt.imshow(slices, cmap=plt.cm.gray, vmin=0, vmax=2.5)
        ax = plt.gca()
        ax.axis("off")
        ax.invert_yaxis()
        plt.savefig("figure4.eps", dpi=300)

    def plot_slice_diff(
        self,
        dim: str = "z",
        type_bg: str = "orig",
        type_fg: str = "noised",
        num: str = "32",
        caxis: tuple = (
            None,
            None,
        ),
    ):
        """
        plot_slice_diff

        plots difference in values between two different brains

        Parameters
        ----------
        dim : str, optional
            as above, by default 'z'
        type_bg : str, optional
            background image, by default 'orig'
        type_fg : str, optional
            foreground image, by default 'noised'
        num : str, optional
            as above, by default '32'
        caxis : tuple, optional
            as above, by default (None, None,)
        """
        os.makedirs(self.output_path, exist_ok=True)
        ax = plt.subplot(111)
        img = self.brains[type_fg] - self.brains[type_bg]
        plotting.plot_anat(
            img,
            axes=ax,
            display_mode=dim,
            cut_coords=[num],
            annotate=False,
            draw_cross=False,
            colorbar=True,
            threshold=1e-6,
            # cmap=plt.cm.hot,
            vmin=caxis[0],
            vmax=caxis[1],
        )
        f_name = (
            f"{self.output_path}/{self.rid}_{dim}_"
            f"{str(num).zfill(3)}_{type_fg}_minus_{type_bg}.eps"
        )
        print(ax.properties)
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.close()

    def plot_3d(self):
        raise NotImplementedError
        cutoff_val = 0.2
        mask = self.brains["orig"] > cutoff_val
        # scale colors of the array so they lie between 0 and 1
        colors = np.where(mask, "#FFD65DC0", "#7A88CCC0")
        box = bbox(mask)
        mask = apply_bbox(mask, box)
        surf_mask = surface_mask(mask)
        cmap = plt.cm.Greys(surf_mask)
        cmap[..., -1] = 0.5
        ax = plt.figure().add_subplot(projection="3d")
        ax.view_init(elev=30, azim=45, roll=0)
        ax.grid(False, which="both", axis="both")
        plt.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.voxels(surf_mask, facecolors=cmap)
        plt.savefig("test.svg")


class NiftiBrain:
    """
     Essentially a dataclass that stores a nifti data and its affine
     matrix and shape. Classmethod to create a brain from a .nii file,
     or you can initialize with a loaded Nifti1Image.

    _extended_summary_
    """

    def __init__(self, img: nib.Nifti1Image):
        self.original_brain = img.get_fdata()
        self.brain_img = img
        self.affine = img.affine
        self.shape = self.brain_img.shape

    @classmethod
    def from_file(cls, root: str, folder: str, file_name: str):
        """
        from_file

        creates brain from file

        Parameters
        ----------
        root : str
            Root folder
        folder : str
            subfolder
        file_name : str
            .nii filename

        Returns
        -------
        NiftiBrain
        """
        img = nib.load(os.path.join(root, folder, file_name))
        return NiftiBrain(img)

    def rescale(
        self,
        range_=(
            0,
            2.5,
        ),
    ):
        self.original_brain, (m, t, a) = rescale(self.original_brain, range_)
        self.brain_img = nib.Nifti1Image(self.original_brain, self.affine)
        return m, t, a

    def rescale_from_prev(self, m, t, a):
        self.original_brain, (m, t, a) = rescale_from_prev(self.original_brain, m, t, a)
        self.brain_img = nib.Nifti1Image(self.original_brain, self.affine)
        return m, t, a

    def __subtract__(self, brain):
        assert np.all(brain.affine == self.affine) and np.all(brain.shape == self.shape)
        return NiftiBrain(self.original_brain - brain.original_brain, self.affine)


def generate_mask_dir(basedir="/Users/mromano/research/data/rgan_data/Z"):
    """
    generate_mask_dir Thresholds masked brains and saves them in folder

    Thresholds sliced brains and saves them in a new folder, M,
    in same directory as basedir


    Parameters
    ----------
    basedir : str, optional
        _description_, by default '/Users/mromano/research/data/rgan_data/Z'
    """
    new_dir = os.path.join(basedir, "..", "M")
    os.makedirs(new_dir, exist_ok=True)
    for fi in os.listdir(basedir):
        brain = NiftiBrain.from_file(basedir, "", fi)
        mask = np.where(brain.original_brain > 1e-10, 1.0, 0.0)
        img = nib.Nifti1Image(mask, brain.affine)
        nib.save(img, os.path.join(new_dir, fi))


def bbox(mask_: np.ndarray):
    """
    bbox bounding box around max

    function gives x, y, and z bounding box ranges for an nd.array

    Parameters
    ----------
    mask_ : np.ndarray
        binary mask

    Returns
    -------
    tuple of len(2) arrays
        x, y and z min and max for bounding boxes
    """
    x, y, z = np.indices(mask_.shape)
    x_range = [np.min(x[mask_]), np.max(x[mask_])]
    y_range = [np.min(y[mask_]), np.max(y[mask_])]
    z_range = [np.min(z[mask_]), np.max(z[mask_])]
    return x_range, y_range, z_range


def feature_scale(XX: np.ndarray):
    num_ = XX - np.min(XX)
    denom_ = np.max(XX) - np.min(XX)
    return num_ / denom_


def apply_bbox(XX: np.ndarray, bb: tuple):
    """
    apply_bbox applies a bounding box around mask XX

    using a tuple of bounding box indices, clips the image
    to only those values within the bounding box

    Parameters
    ----------
    XX : np.ndarray
        3-d ndarray to be clipped
    bb : tuple
        tuple of len(3) containing
        len(2) lists containing the beginning and end of the
        bounding boxes for each dimension

    Returns
    -------
    _type_
        bounded nd.array
    """
    XX = XX[bb[0][0] : (bb[0][1] + 1), ...]
    XX = XX[:, bb[1][0] : (bb[1][1] + 1), :]
    XX = XX[..., bb[2][0] : (bb[2][1] + 1)]
    return XX


def dist_from_origin(mask_: np.ndarray):
    raise NotImplementedError
    # TODO: computes the distance from the origin
    x, y, z = np.indices(mask_.shape)
    origin = np.array(mask_.shape) // 2
    dist = ((x - origin[0]) ** 2 + (y - origin[1]) ** 2 + (z - origin[2]) ** 2) ** (
        1 / 3
    )
    dist = feature_scale(dist)
    dist[~mask_] = 0
    return dist


# help from https://scikit-image.org/docs/stable/auto_examples/applications/plot_fluorescence_nuclear_envelope.html
def surface_mask(mask_: np.ndarray):
    """
    surface_mask:
    creates a surface mask of a 3-dimensional volume


    Parameters
    ----------
    mask_ : np.ndarray
        3d array mask

    Returns
    -------
    np.ndarray
        masked nd.array
    """
    mask_ = np.where(mask_, 1, 0)
    return np.logical_and(
        morphology.binary_dilation(mask_), not morphology.binary_erosion(mask_)
    )


def plot_slices(
    type_: str = "orig",
    caxis=(
        0,
        2.5,
    ),
):
    """
    plot_slices

    Plots slices for each axis in a given brain at specified coordinates

    Parameters
    ----------
    type_ : str, optional
        type of brain (Mask, Original, Novel, etc), by default 'orig'
    caxis : tuple, optional
        MNI coordinates of slices, by default (0,2.5,)
    """
    num_z = -85
    num_x = 60
    num_y = 85
    bs = BrainSample()
    bs.plot_slice(dim="z", type_=type_, num=num_z, caxis=caxis, colorbar=True)
    bs.plot_slice(dim="z", type_=type_, num=num_z, caxis=caxis, colorbar=False)
    bs.plot_slice(dim="x", type_=type_, num=num_x, caxis=caxis, colorbar=False)
    bs.plot_slice(dim="y", type_=type_, num=num_y, caxis=caxis, colorbar=False)


def output_shape(input_shape, padding, kernel, stride):
    """
    output_shape

    Outputs the resultant shape of a convolutional layer

    Parameters
    ----------
    input_shape : list-like
        shape of input tensor
    padding : int
        padding width (assumed to be isometric)
    kernel : int
        kernel width (assumed to be isometric)
    stride : int
        stride

    Returns
    -------
    np.array
        output shape after a convolutional layer
    """
    input_shape = np.array(input_shape)
    return np.floor(1 + (input_shape + 2 * padding * (kernel - 1) - 1) / stride)


if __name__ == "__main__":
    # generate_mask_dir()
    bs = BrainSample()
    bs.plot_missing_montage()
    caxis = (
        0,
        2.5,
    )
    plot_slices("orig", caxis)
    plot_slices("noised")
    plot_slices("novel")
    plot_slices("vanilla")
    plot_slices("mask")
    bs.plot_slice_diff(
        dim="z", type_bg="vanilla", type_fg="novel", num=-85, caxis=(-1.5, 1.5)
    )
