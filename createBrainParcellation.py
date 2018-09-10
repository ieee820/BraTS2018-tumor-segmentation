import argparse
import os
from nipype.interfaces.ants import N4BiasFieldCorrection
import subprocess
from natsort import natsorted
import SimpleITK as sitk
import numpy as np

def ReadImage(file_path):
    ''' This code returns the numpy nd array for a MR image at path'''
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float32)

def N4ITK(filepath, output_name):
    print('N4ITK working on: %s' %filepath) 
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = filepath
    n4.inputs.output_image = output_name
    n4.run()

def RegisterBrain(t1_path, ref_path, subject2mni_mat, mni2subject_mat):
    print('Working on registration!')
    # Create the affine transformation matrix from subject space to MNI152 1mm space
    subprocess.call(["flirt", "-in", t1_path, "-ref", ref_path, "-omat", subject2mni_mat])
    subprocess.call(["convert_xfm", "-omat", mni2subject_mat, "-inverse", subject2mni_mat])
    print('Finish this subject!')

def RegisterLabels2Subject(refVol_path, bp_filepaths, mni2subject_mat):
	''' register indivudial labels from MNI 152 space to subject space '''
	for j in range(len(bp_filepaths)):
		label_name = "./lab"+str(j+1)+".nii.gz"
		# Register Brain Labels to Subject Space
		subprocess.call(["flirt", "-in", bp_filepaths[j], "-ref", refVol_path, "-out", label_name, "-init", mni2subject_mat, "-applyxfm"])

def SubjectLabels2ParcellationArgmax(subject_bp_filepaths, subject_name):
	print('Mapping brain parcellation to subject')
	subjectBrainParcellations = np.zeros((len(subject_bp_filepaths)+1, 155, 240, 240), dtype=np.float32)
	img = sitk.ReadImage(subject_bp_filepaths[0])
	for j, bp in enumerate(subject_bp_filepaths):
		subjectBrainParcellations[j+1,:] = ReadImage(bp)
	brainParcellation = np.argmax(subjectBrainParcellations, axis=0)
	brainParcellationFloat = brainParcellation.astype(np.float32)
	brainParcellationFloat_img = sitk.GetImageFromArray(brainParcellationFloat)
	brainParcellationFloat_img.CopyInformation(img)
	sitk.WriteImage(brainParcellationFloat_img, subject_name)
def Remove(filepaths):
	for file in filepaths:
		os.remove(file)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="the input directory to a BraTS subject T1 image", type=str)
parser.add_argument("-o", "--output", help="the output directory to save the subject's brain parcellation", type=str)
parser.add_argument("-n", "--name", help="subject name", type=str)
args = parser.parse_args()

# N4ITK
filepath = args.input
N4ITK_path = filepath[:filepath.find(".nii.gz")]+'_temp.nii.gz'
# Apply bias correction on MR T1 image
N4ITK(filepath, N4ITK_path)

# Registration
mni152_1mm_path = './MNI152_T1_1mm_brain.nii.gz'
subject2mni_path = filepath[:filepath.index('.nii.gz')]+'_invol2refvol.mat'
mni2subject_path = filepath[:filepath.index('.nii.gz')]+'_refvol2invol.mat'
RegisterBrain(N4ITK_path, mni152_1mm_path, subject2mni_path, mni2subject_path)
brain_parcellation_path = './atlases/HarvardOxford'

# Mapping individual brain parcellation to subject
bp_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(brain_parcellation_path) for name in files if name.endswith('.nii.gz')]
bp_filepaths = natsorted(bp_filepaths, key=lambda y: y.lower())
RegisterLabels2Subject(filepath, bp_filepaths, mni2subject_path)

# Merge individual labels to the brain parcellation in subject space using argmax
subject_bp_filepaths = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.split(filepath)[0]) for name in files if 'HarvardOxford' not in name and 'lab' in name and name.endswith('.nii.gz')]
subject_bp_filepaths = natsorted(subject_bp_filepaths, key=lambda y: y.lower())
subject_name = os.path.join(os.path.split(filepath)[0], args.name+'_HarvardOxford-sub.nii.gz')
SubjectLabels2ParcellationArgmax(subject_bp_filepaths, subject_name)

Remove(subject_bp_filepaths)
os.remove(N4ITK_path)
os.remove(subject2mni_path)
os.remove(mni2subject_path)
