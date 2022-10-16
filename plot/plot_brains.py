from ast import Not
import os
import subprocess
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from dataclasses import dataclass

def rescale(array, tup):
    m = np.min(array)
    if m < 0:
        array += -m
    a = np.max(array)-np.min(array)
    t = tup[1] - tup[0]
    return (array * t / a), (m, t, a)

def rescale_from_prev(array, m, t, a):
    if m < 0:
        array += -m
    return array * t / a, (m, t, a)

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
    prefix = 'masked_brain_mri_'
    file_type = '.nii'
    folder_map = {'T': 'orig', 'Z': 'noised', 'G': 'vanilla', 'CG_1': 'novel', 'M': 'mask'}

    def __init__(self, brain_path='/Users/mromano/research/data/rgan_data', rid='0307'):
        self.rid = rid
        self.path = brain_path
        self.output_path = os.path.join(brain_path, '..', 'img')
        self.brains = {}
        for f, nm in self.folder_map.items():
            self.brains[nm]  = Brain(brain_path, f, self.prefix + self.rid + self.file_type)

    def plot_slice(
        self, 
        dim: str = 'z', 
        type_: str = 'orig', 
        num: str = '32', 
        threshold=1e-06, 
        caxis: tuple=(0, 2.5),
        colorbar: bool=False):
        os.makedirs(self.output_path, exist_ok=True)
        ax = plt.subplot(111)
        plotting.plot_anat(self.brains[type_].brain_img,
                            axes=ax,
                            display_mode=dim,
                            cut_coords=[num],
                            threshold=threshold,
                            annotate=False,
                            vmax=caxis[1],
                            vmin=caxis[0],
                            colorbar=colorbar,
                            black_bg=False
                            )
        f_name = f'{self.output_path}/{self.rid}_{dim}_' \
                    f'{str(num).zfill(3)}_{type_}.eps'
        if colorbar:
            print(ax.)
            raise NotImplementedError
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.close()

    def plot_slice_overlay(
        self,
        dim: str = 'z', 
        type_bg: str = 'orig', 
        type_fg: str = 'noised', 
        num: str = '32',
        threshold: float=1e-06,
        colorbar=False):
        os.makedirs(self.output_path, exist_ok=True)
        ax = plt.subplot(111)
        plotting.plot_stat_map(
            self.brains[type_fg].brain_img,
            bg_img=self.brains[type_bg].brain_img,
            axes=ax,
            display_mode=dim,
            cut_coords=[num],
            annotate=False,
            draw_cross=False,
            threshold=threshold,
            vmax=caxis[1],
            vmin=caxis[0],
            colorbar=colorbar
            )
        f_name = f'{self.output_path}/{self.rid}_{dim}_' \
                    f'{str(num).zfill(3)}_{type_fg}_over_{type_bg}.eps'
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.close()

    def plot_slice_diff(
        self,
        dim: str = 'z', 
        type_bg: str = 'orig', 
        type_fg: str = 'noised', 
        num: str = '32',
        caxis: tuple = (None, None,)
        ):
        os.makedirs(self.output_path, exist_ok=True)
        ax = plt.subplot(111)
        img = nib.Nifti1Image(
            self.brains[type_fg].original_brain-self.brains[type_bg].original_brain,
            affine=self.brains[type_bg].affine
        )
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
            vmax=caxis[1]
            )
        f_name = f'{self.output_path}/{self.rid}_{dim}_' \
                    f'{str(num).zfill(3)}_{type_fg}_minus_{type_bg}.eps'
        print(ax.properties)
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.close()

    def plot_3d(self,
        dim: str = 'z', 
        type_bg: str = 'orig', 
        type_fg: str = 'noised', 
        num: str = '32',
        caxis: tuple = (None, None,)
        ):
        np.imshow(self.)

@dataclass
class Brain:
    root: str
    folder: str
    file_name: str

    def __post_init__(self):
        data = nib.load(os.path.join(self.root, self.folder, self.file_name))
        self.original_brain = data.get_fdata()  
        print(np.max(self.original_brain))
        self.brain_img = data
        self.affine = data.affine

    def rescale(self, range_=(0, 2.5,)):
        self.original_brain, (m, t, a) = rescale(self.original_brain, range_)
        self.brain_img = nib.Nifti1Image(self.original_brain, self.affine)
        return m, t, a
    
    def rescale_from_prev(self, m, t, a):
        self.original_brain, (m, t, a) = rescale_from_prev(self.original_brain, m, t, a)
        self.brain_img = nib.Nifti1Image(self.original_brain, self.affine)
        return m, t, a

def generate_mask_dir(basedir='/Users/mromano/research/data/rgan_data/Z'):
    new_dir = os.path.join(basedir,'..','M')
    os.makedirs(new_dir, exist_ok=True)
    for fi in os.listdir(basedir):
        brain = Brain(basedir, '', fi)
        mask = np.where(brain.original_brain > 1e-10,1.,0.)
        img = nib.Nifti1Image(mask, brain.affine)
        nib.save(img, os.path.join(new_dir, fi))

if __name__ == '__main__':
    bs = BrainSample()
    num = -85
    caxis = (0, 2.5)
    # generate_mask_dir()
    # # slice -- want the hippocampus
    # m, t, a = bs.brains['orig'].rescale()
    # m, t, a = bs.brains['noised'].rescale_(m, t, a)
    bs.plot_slice(dim='z', type_='orig', num=num, caxis=caxis, colorbar=True)
    bs.plot_slice(dim='z', type_='noised', num=num, caxis=caxis)
    bs.plot_slice(dim='z', type_='vanilla', num=num, caxis=caxis)
    bs.plot_slice(dim='z', type_='novel', num=num, caxis=caxis)
    bs.plot_slice(dim='z', type_='mask', num=num, caxis=(0,1,))
    # bs.plot_slice_overlay(dim='z', type_bg='orig', type_fg='mask', num=num)
    bs.plot_slice_diff(dim='z', type_bg='vanilla', type_fg='novel', num=num, caxis=(-1.5,1.5))