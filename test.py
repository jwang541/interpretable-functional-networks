import scipy
import nibabel as nib



data1 = nib.load('resources/t1/fn15.nii').get_fdata()
data2 = nib.load('resources/t400/fn15.nii').get_fdata()

print(data1.shape)
print(data2.shape)

data1 = data1.flatten()
data2 = data2.flatten()

print(data1.shape)
print(data2.shape)

print(scipy.stats.pearsonr(data1, data2))