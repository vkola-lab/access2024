import os
import random
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import plotting
from skimage import filters, morphology
from tqdm import tqdm


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


def feature_scale(array) -> np.array:
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
    a = np.max(array) - m
    return (array - m) / a


def sobel_filter(array) -> np.ndarray:
    return filters.sobel(array)


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
    """
    Reads "slice" type files; returns list
    """
    with open(fname, "r") as fi:
        str_ = fi.readlines()
        return eval(str_[0])


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
        # "linear_interpolation": "linear",
    }
    mask_rel_path = "background_masks/wc{layer}{rid}_mri.nii"
    idx_rel_path = "slice_list/ADNI"

    def __init__(
        self, brain_path="/Users/mromano/research/data/rgan_data", rid="0307"
    ):
        self.rid = rid
        self.path = brain_path
        self.output_path = os.path.join(brain_path, "..", "img")
        self.brains = {}
        for f, nm in self.folder_map.items():
            self.brains[nm] = NiftiBrain.from_file(
                brain_path, f, self.prefix + self.rid + self.file_type
            )
        self._load_masks()

        self.idx_missing = read_txt(
            os.path.join(
                brain_path, self.idx_rel_path, self.prefix + self.rid + ".txt"
            )
        )
        self.idx_missing.sort()

    def _load_masks(self) -> None:
        background_mask_1 = NiftiBrain.from_file(
            self.path, "", self.mask_rel_path.format(layer="1", rid=rid)
        )
        background_mask_2 = NiftiBrain.from_file(
            self.path, "", self.mask_rel_path.format(layer="2", rid=rid)
        )
        background_mask_3 = NiftiBrain.from_file(
            self.path, "", self.mask_rel_path.format(layer="3", rid=rid)
        )
        brain_sum = background_mask_1 + background_mask_2 + background_mask_3
        self.background_mask = brain_sum.threshold(0.2)
        self.background_mask.original_brain = (
            1 - self.background_mask.original_brain
        )
        self.background_mask.affine = self.brains["orig"].affine

        self.ventricle_mask = background_mask_3.threshold(0.2)
        self.ventricle_mask.affine = self.brains["orig"].affine

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
            f"{str(num).zfill(3)}_{type_}_colorbar_{str(colorbar)}.png"
        )
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.close()

    def plot_missing_montage(self, output_fi: str = "", normalize=False):
        """
        plot_missing_montage _summary_

        _extended_summary_
        """
        slices = []
        if output_fi == "":
            output_fi = "figure3_" + str(self.rid) + ".eps"
        print(self.mask.shape)
        print(self.brains["orig"].shape)

        for idx in np.arange(5, len(self.idx_missing) - 5, 10):
            curr_slices = []
            for val in ["orig", "vanilla", "novel"]:
                if normalize:
                    self.brains[val].feature_scale()
                curr_slices.append(
                    np.squeeze(
                        self.brains[val].original_brain[
                            self.idx_missing[idx], :, :
                        ]
                    )
                )
            curr_slices = np.concatenate(curr_slices, axis=1).T
            slices.append(curr_slices)
        slices = np.concatenate(slices, axis=1)
        ax = plt.imshow(slices, cmap=plt.cm.gray)
        ax = plt.gca()
        ax.axis("off")
        ax.invert_yaxis()
        plt.colorbar()
        plt.savefig(output_fi, dpi=300)
        plt.close()

    def plot_missing_montage_masked(
        self, output_fi: str = "", normalize=False
    ):
        """
        plot_missing_montage _summary_

        _extended_summary_
        """
        slices = []
        ventricle_slices = []
        background_slices = []
        if output_fi == "":
            output_fi = "figure3_" + str(self.rid) + ".eps"

        for idx in np.arange(15, len(self.idx_missing) - 5, 10):
            curr_slices = []
            curr_ventricle_slices = []
            curr_background_slices = []
            for val in ["orig", "vanilla", "novel"]:
                if normalize:
                    self.brains[val].feature_scale()
                curr_slices.append(
                    np.squeeze(
                        self.brains[val].original_brain[
                            self.idx_missing[idx], :, :
                        ]
                    )
                )
                curr_ventricle_slices.append(
                    np.squeeze(
                        self.ventricle_mask.original_brain[
                            self.idx_missing[idx], :, :
                        ]
                    )
                )
                curr_background_slices.append(
                    np.squeeze(
                        self.background_mask.original_brain[
                            self.idx_missing[idx], :, :
                        ]
                    )
                )
            curr_slices = np.concatenate(curr_slices, axis=1).T
            curr_ventricle_slices = np.concatenate(
                curr_ventricle_slices, axis=1
            ).T
            curr_background_slices = np.concatenate(
                curr_background_slices, axis=1
            ).T

            slices.append(curr_slices)
            ventricle_slices.append(curr_ventricle_slices)
            background_slices.append(curr_background_slices)

        slices = np.concatenate(slices, axis=1)
        ventricle_slices = np.concatenate(ventricle_slices, axis=1)
        background_slices = np.concatenate(background_slices, axis=1)

        ax = plt.imshow(slices, cmap=plt.cm.gray, alpha=1)
        ventricle_slices[np.where(ventricle_slices == 0)] = np.nan
        background_slices[np.where(background_slices == 0)] = np.nan

        plt.imshow(ventricle_slices, alpha=1.0, cmap=ListedColormap(["blue"]))
        plt.imshow(
            background_slices, alpha=1.0, cmap=ListedColormap(["black"])
        )
        ax = plt.gca()
        ax.axis("off")
        ax.invert_yaxis()
        plt.title("Subject " + str(self.rid))
        plt.savefig(output_fi, dpi=300)
        plt.close()

    def plot_brain_hist(self) -> None:
        """
        plot_missing_montage _summary_

        _extended_summary_
        """
        ax = plt.subplot()
        for _, val in enumerate(["orig", "vanilla", "novel"]):
            ax.hist(self.brains[val].original_brain.reshape(-1, 1), bins=50)
            ax.set_title(val)
            ax.set_ylim([0, 1.6e6])
        plt.savefig("plot/histograms.png", dpi=300)
        plt.close()


class NiftiBrain:
    """
     Essentially a dataclass that stores a nifti data and its affine
     matrix and shape. Classmethod to create a brain from a .nii file,
     or you can initialize with a loaded Nifti1Image.

    _extended_summary_
    """

    def __init__(self, img: nib.Nifti1Image) -> None:
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

    def feature_scale(
        self,
    ):
        self.original_brain = feature_scale(self.original_brain)
        self.brain_img = nib.Nifti1Image(self.original_brain, self.affine)

    def edge_detect(
        self,
    ):
        filtered_brain = sobel_filter(self.original_brain)
        brain_img = nib.Nifti1Image(filtered_brain, self.affine)
        return NiftiBrain(brain_img)

    def rescale_from_prev(self, m, t, a):
        self.original_brain, (m, t, a) = rescale_from_prev(
            self.original_brain, m, t, a
        )
        self.brain_img = nib.Nifti1Image(self.original_brain, self.affine)
        return m, t, a

    def threshold(self, val: int = 0):
        original_brain_threshold = np.where(self.original_brain > val, 1.0, 0)
        return NiftiBrain(
            nib.Nifti1Image(original_brain_threshold, self.affine)
        )

    def __sub__(self, brain):
        assert np.all(brain.affine == self.affine) and np.all(
            brain.shape == self.shape
        )
        return NiftiBrain(
            nib.Nifti1Image(
                self.original_brain - brain.original_brain, self.affine
            )
        )

    def __add__(self, brain):
        assert np.all(brain.affine == self.affine) and np.all(
            brain.shape == self.shape
        )
        return NiftiBrain(
            nib.Nifti1Image(
                self.original_brain + brain.original_brain, self.affine
            )
        )


def plot_slices(
    type_: str = "orig",
    caxis=(
        0,
        2.5,
    ),
    rid="0307",
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
    bs = BrainSample(rid=rid)
    bs.plot_slice(dim="z", type_=type_, num=num_z, caxis=caxis, colorbar=True)
    bs.plot_slice(dim="z", type_=type_, num=num_z, caxis=caxis, colorbar=False)
    bs.plot_slice(dim="x", type_=type_, num=num_x, caxis=caxis, colorbar=False)
    bs.plot_slice(dim="y", type_=type_, num=num_y, caxis=caxis, colorbar=False)


def plot_sample_slices_all():
    files = os.listdir("/Users/mromano/research/data/rgan_data/Z")
    rids = list(map(lambda x: re.sub("masked_brain_mri_", "", x)[:-4], files))
    for rid in tqdm(rids):
        bs_ = BrainSample(rid=rid)
        bs_.plot_missing_montage(
            output_fi=f"/Users/mromano/research/data/img/sample_slices_{rid}_z.png"
        )
        # bs_.plot_slice(
        #     type_="linear",
        # )


def _sample_rids(n_samples: int) -> list[str]:
    fi = os.listdir("/Users/mromano/research/data/rgan_data/Z/")
    reg = re.compile(r".*_(?P<rid>[0-9]{4})\.nii")
    rids = list(map(lambda x: reg.match(x)["rid"], fi))
    random.seed(1)
    return random.sample(rids, n_samples)


if __name__ == "__main__":
    # generate_mask_dir()
    rid_list = _sample_rids(8)
    for rid in rid_list:
        bs = BrainSample(rid=rid)
        # bs.plot_brain_hist()
        bs.plot_missing_montage_masked(normalize=True)
        # caxis = (
        #     0,
        #     2.5,
        # )
        # plot_slices("orig", caxis)
        # plot_slices("noised")
        # plot_slices("novel")
        # plot_slices("vanilla")
        # # plot_slices("linear")
        # bs.plot_slice_diff(
        #     dim="z", type_bg="vanilla", type_fg="novel", num=-85, caxis=(-1.5, 1.5)
        # )
        # bs.plot_slice_edge(dim="z", type_="vanilla", num=-85, caxis=(-1.5, 1.5))
        # plot_sample_slices_all()
