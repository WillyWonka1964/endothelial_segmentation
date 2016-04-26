import numpy as np
import pandas as pd
import skimage
import re
import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from os import listdir
from os.path import isfile, join
from scipy import ndimage as ndi
from scipy.misc import bytescale
from scipy.spatial.distance import pdist, squareform
from skimage import measure, exposure, morphology, img_as_uint, img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from skimage.filters import *
from skimage.io import imsave
from skimage.morphology import disk, watershed, remove_small_objects
from skimage.segmentation import *
from sklearn.externals import joblib
from sklearn import preprocessing

start_time = datetime.now()

file_path = '../01_TIF/'

pixel_dimension = 0.1612 #micro meter

nucleus_channel = 1
plasmaMembrane_channel = 3
wpb_channel = 2
golgi_channel = None
exitSites_channel = None

nucleus_markers_sensitivity = 20

wpb_block_size = 21 #adaptive threshold block size
wpb_offset = -1 #adaptive threshold offset
wpb_min_area = 5 #pixels
wpb_max_area = 385 #pixels

golgi_block_size = 20 #adaptive threshold block size
golgi_offset = -10 #adaptive threshold offset
golgi_min_area = 5 #pixels
golgi_max_area = 2000 #pixels

exit_sites_threshold = 796 #16-bit
exit_sites_min_area = 5 #pixels

regex=re.compile("") # list images to exclude as ".*(003004001_Field_001|006006001_Field_005).*"

image_list = set([f for f in listdir(file_path) if isfile(join(file_path,f))])
exclude_list = set([m.group(0) for l in image_list for m in [regex.search(l)] if m])
image_list = sorted(image_list - exclude_list)#[51:56]
number_of_images = len(image_list)
experiment_name  = os.path.basename(os.path.dirname(os.path.dirname(os.getcwd())))

scaler_mean = np.array([9.96611432e+01,   3.24928525e+01,   1.14824547e+01, 8.82224464e+00,   1.03498904e+02,   1.12612778e+01, 7.05924323e+00,   1.71932105e-01,   9.71748933e-01, 7.60923639e+02,   2.88836940e+02,   4.52846104e+02])
scaler_std = np.array([9.72755097e+01,   2.62195042e+01,   8.81500109e+00, 7.00432068e+00,   1.03254443e+02,   8.84043825e+00, 5.85617067e+00,   8.79274776e-01,   3.63298460e-02, 4.61973492e+02,   4.06499218e+01,   1.62126888e+02])

channels = [nucleus_channel, plasmaMembrane_channel, wpb_channel, golgi_channel, exitSites_channel]
single_cell_analysis = False
if type(plasmaMembrane_channel) == int: single_cell_analysis = True

def scale8bit(image):
    scale = float(256) / (image.max() - image.min())
    return np.clip(np.round(np.multiply(image, scale)), 0, 255).astype(np.uint8)

def calculateFeret(coordinates):
    feret = np.nanmax(squareform(pdist(coordinates)))
    feret = feret + (((2*((0.5)**2))**(0.5))*2)
    return feret

def nucleusSegmentation(nucleus_image, nucleus_markers_sensitivity):
    nucleus_rescale = rescale_intensity(nucleus_image, in_range=(0,nucleus_image.max()))
    otsu_thresh = threshold_otsu(nucleus_image)
    blobs = nucleus_image > otsu_thresh
    labels = skimage.measure.label(blobs)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    blobs[labels != background] = 255
    distance = ndi.distance_transform_edt(blobs)
    blobs_labels = watershed(-distance, ndi.label(remove_small_objects(distance > nucleus_markers_sensitivity, min_size=10))[0], mask=blobs)
    blobs = blobs^find_boundaries(blobs_labels, mode='inner')
    blobs_labels = ndi.label(blobs)[0]
    nucleus_features = measureMorphometry(blobs_labels, nucleus_image, image_name)
    nucleus_features_scaled = ((nucleus_features.drop(nucleus_features.columns[[0,1,2,3,4,5]], axis=1) - scaler_mean)/scaler_std)
    svm_classifier = joblib.load('../../nucleus_svm_model.pkl')
    svm_prediction = np.asarray(np.where(svm_classifier.predict(nucleus_features_scaled)==0)) + 1
    for x in range(0,svm_prediction.shape[1]): blobs_labels[blobs_labels==svm_prediction[0,x]] = 0
    blobs = blobs_labels > 0
    blobs_labels = ndi.label(blobs)[0]
    nucleus_features = measureMorphometry(blobs_labels, nucleus_image, image_name)
    contours = find_boundaries(blobs_labels, mode='outer').astype(np.uint8)*255
    boundaries = mark_boundaries(nucleus_rescale, blobs_labels, color=(0.6,0,0), outline_color=None, mode='outer')
    imsave('../03_python_results/nucleus_overlay/'+image_name.split('.',1)[0]+'.png', boundaries)
    return (blobs, blobs_labels, contours, boundaries, nucleus_features)

def golgiSegmentation(golgi_image):
    binary = threshold_adaptive(golgi_image, golgi_block_size, offset=golgi_offset)
    blobs = remove_small_objects(binary, min_size=golgi_min_area) - remove_small_objects(binary, min_size=golgi_max_area)
    blobs_labels = ndi.label(blobs)[0]
    boundaries = mark_boundaries(scale8bit(golgi_image), blobs_labels, color=(0.6,0,0), outline_color=None, mode='outer')
    golgi_features = measureMorphometry(blobs_labels, golgi_image, image_name)
    imsave('../03_python_results/golgi_overlay/'+image_name.split('.',1)[0]+'.png', boundaries)
    return (blobs, blobs_labels, boundaries, golgi_features)

def weibelPaladeSegmentation(wpb_image):
    wpb_image = img_as_ubyte(wpb_image)
    binary = threshold_adaptive(wpb_image, wpb_block_size, offset=wpb_offset)
    blobs = remove_small_objects(binary, min_size=wpb_min_area) - remove_small_objects(binary, min_size=wpb_max_area)
    blobs_labels = ndi.label(blobs)[0]
    contours = find_boundaries(blobs_labels, mode='outer').astype(np.uint8)*255
    boundaries = mark_boundaries(scale8bit(wpb_image), blobs_labels, color=(0.6,0,0), outline_color=None, mode='outer')
    wpb_features = measureMorphometry(blobs_labels, wpb_image, image_name)
    imsave('../03_python_results/wpb_overlay/'+image_name.split('.',1)[0]+'.png', boundaries)
    return (blobs, blobs_labels, contours, boundaries, wpb_features)

def exitSitesSegmentation(exitSites_image, threshold_value, exit_sites_min_area):
    blur = skimage.img_as_int(skimage.filters.gaussian_filter(exitSites_image, sigma=1))
    threshold = blur; threshold[threshold < threshold_value] = 0
    binary = threshold > exit_sites_threshold
    blobs = remove_small_objects(~(remove_small_objects(~binary, min_size=exit_sites_min_area)), min_size=exit_sites_min_area)
    local_maxi = peak_local_max(blur, min_distance=5, footprint=np.ones((9, 9)), labels=blobs, exclude_border=True, indices=False)
    blobs_labels = watershed(-blur, ndi.label(local_maxi)[0], mask=blobs)
    blobs_labels = morphology.remove_small_objects(blobs_labels, min_size=exit_sites_min_area)
    boundaries = mark_boundaries(scale8bit(exitSites_image), blobs_labels, color=(0.6,0,0), outline_color=None, mode='outer')
    exit_sites_features = measureMorphometry(blobs_labels, exitSites_image, image_name)
    imsave('../03_python_results/exit_sites_overlay/'+image_name.split('.',1)[0]+'.png', boundaries)
    return (blobs, blobs_labels, boundaries, exit_sites_features)

def cellSegmentation(plasmaMembrane_image, nucleus_blobs):
    plasmaMembrane_image = skimage.exposure.equalize_adapthist(plasmaMembrane_image, ntiles_x=20, clip_limit=0.01)
    plasmaMembrane_denoise = skimage.restoration.denoise_tv_chambolle(plasmaMembrane_image, weight=0.1)
    plasmaMembrane_seeds = np.fmin(~img_as_uint(nucleus_blobs), img_as_uint(plasmaMembrane_denoise))
    nucleus_markers = morphology.dilation((ndi.label(nucleus_blobs)[0]), disk(20))
    blobs_labels = watershed(plasmaMembrane_seeds, nucleus_markers)
    blobs = clear_border(~np.zeros(blobs_labels.shape, dtype=bool)^find_boundaries(blobs_labels, mode='inner'))
    blobs_labels = ndi.label(blobs)[0]
    contours = find_boundaries(blobs_labels, mode='outer').astype(np.uint8)*255
    boundaries = mark_boundaries(plasmaMembrane_image, blobs_labels, color=(0.6,0,0), outline_color=None, mode='thick')
    cell_features = measureMorphometry(blobs_labels, blobs, image_name)
    imsave('../03_python_results/cell_overlay/'+image_name.split('.',1)[0]+'.png', boundaries)
    imsave('../03_python_results/cell_labels/'+image_name.split('.',1)[0]+'.png', blobs_labels)
    return (blobs, blobs_labels, contours, boundaries, cell_features)

def measureMorphometry(label_image, intensity_image, image_name):
    properties = measure.regionprops(label_image, intensity_image)
    properties_boundary = measure.regionprops(find_boundaries(label_image, mode='thick')*label_image)
    y_centroid = pd.Series([i[0] for i in [prop.centroid for prop in properties]]) * pixel_dimension
    x_centroid = pd.Series([i[1] for i in [prop.centroid for prop in properties]]) * pixel_dimension
    area = pd.Series([prop.area for prop in properties]) * pow(pixel_dimension, 2)
    perimeter = pd.Series([prop.perimeter for prop in properties]) * pixel_dimension
    feret = pd.Series([calculateFeret(prop.coords) for prop in properties_boundary]) * pixel_dimension
    equivalent_diameter = pd.Series([prop.equivalent_diameter for prop in properties]) * pixel_dimension
    convex_area = pd.Series([prop.convex_area for prop in properties]) * pow(pixel_dimension, 2)
    major_axis_length = pd.Series([prop.major_axis_length for prop in properties]) * pixel_dimension
    minor_axis_length = pd.Series([prop.minor_axis_length for prop in properties]) * pixel_dimension
    orientation = pd.Series([prop.orientation for prop in properties])
    solidity = pd.Series([prop.solidity for prop in properties])
    max_intensity = pd.Series([prop.max_intensity for prop in properties])
    min_intensity = pd.Series([prop.min_intensity for prop in properties])
    mean_intensity = pd.Series([prop.mean_intensity for prop in properties])
    particles_image = pd.concat([x_centroid,y_centroid,area,perimeter,feret,equivalent_diameter,convex_area,major_axis_length,minor_axis_length,orientation,solidity,max_intensity,min_intensity,mean_intensity],axis=1)
    particle_id = "%09.0f" % float(re.sub("001_Field_", '',image_name.split('.',1)[0]))
    row =  pd.DataFrame([(particle_id[0:3].lstrip('0'))]*particles_image.shape[0])
    col =  pd.DataFrame([(particle_id[3:6].lstrip('0'))]*particles_image.shape[0])
    fov =  pd.DataFrame([(particle_id[6:9].lstrip('0'))]*particles_image.shape[0])
    particle_id = pd.DataFrame([particle_id]*particles_image.shape[0])
    pd.concat([particle_id,row,col,fov,particles_image], axis=1)
    return pd.concat([particle_id,row,col,fov,particles_image], axis=1, ignore_index=True)

def assignCell(label_image, intensity_image, features):
    properties = measure.regionprops(label_image, intensity_image)
    cell = pd.Series([prop.max_intensity for prop in properties])
    features['cell'] = cell
    cols = features.columns.tolist()
    cols.insert(4, cols.pop())
    features = features[cols]
    features[0] = features[0].map(str) + features['cell'].map("{:03}".format).map(str)
    return features

def create_composite(channel_1, channel_2, channel_3, channel_4, contours):
    img_dim = tiffStack.shape[1:]
    img = np.zeros((img_dim[0], img_dim[1], 3), dtype=np.float)
    img[:,:,0] = channel_1 + contours
    img[:,:,1] = channel_2 + contours
    img[:,:,2] = channel_3 + contours
    img[:,:,1] = channel_4 + contours
    img[:,:,2] = channel_4 + contours
    p2, p98 = np.percentile(img, (2, 95))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return bytescale(img_rescale)

def syntheticCoordinates(cell_labels, features):
    properties = measure.regionprops(cell_labels)
    coordinates =[prop.coords for prop in properties]
    coords = np.empty([features.shape[0], 2])
    coords.fill(np.nan)
    for index, row in features.iterrows():
        cell_number = int(features['cell'][index])
        if (cell_number > 0):
            random_coordinates = coordinates[cell_number-1][np.random.randint(0,coordinates[cell_number-1].shape[0],1)]
            coords[index,0] = random_coordinates[0,1]
            coords[index,1] = random_coordinates[0,0]
    if (features.shape[0] > 0):
        features['x_synth']=pd.DataFrame(coords[:,0])*pixel_dimension
        features['y_synth']=pd.DataFrame(coords[:,1])*pixel_dimension

cols = ['particle_id','row','col','fov','x_centroid', 'y_centroid','area','perimeter','feret','equivalent_diameter','convex_area','major_axis_length','minor_axis_length','orientation','solidity','max_intensity','min_intensity','mean_intensity']
if single_cell_analysis == True: cols.insert(4, 'cell')

if type(nucleus_channel) == int: pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_nuclei.csv', sep=',', header=True, index=False)
if type(plasmaMembrane_channel) == int: pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_cells.csv', sep=',', header=True, index=False)
if type(wpb_channel) == int: pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_wpb.csv', sep=',', header=True, index=False)
if type(golgi_channel) == int: pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_golgi.csv', sep=',', header=True, index=False)
if type(exitSites_channel) == int: pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_exit_sites.csv', sep=',', header=True, index=False)

#pd.DataFrame(columns = cols+['x_synth','y_synth']).to_csv('../03_python_results/'+experiment_name+'_wpb.csv', sep=',', header=True, index=False)

for image in range(0, number_of_images):
    image_name = image_list[image]
    tiffStack = skimage.io.imread(file_path+image_name, plugin='tifffile')

    print "Analysing {0}, image {1} of {2}, detected:".format(image_name, image+1, number_of_images)

    if type(nucleus_channel) == int:
        nucleus_image = tiffStack[nucleus_channel-1]
        nucleus_blobs, nucleus_labels, nucleus_contours, nucleus_boundaries, nucleus_features = nucleusSegmentation(nucleus_image, nucleus_markers_sensitivity)
        print "    {0} nuclei ".format(len(nucleus_features))
        if single_cell_analysis == False: nucleus_features.to_csv('../03_python_results/'+experiment_name+'_nuclei.csv', sep=',', mode='a', header=False, index=False)
    if type(plasmaMembrane_channel) == int:
        plasmaMembrane_image = tiffStack[plasmaMembrane_channel-1]
        cell_blobs, cell_labels, cell_contours, cell_boundaries, cell_features = cellSegmentation(plasmaMembrane_image, nucleus_blobs)
        print "    {0} cells ".format(len(cell_features))
        if single_cell_analysis == False: cell_features.to_csv('../03_python_results/'+experiment_name+'_cells.csv', sep=',', mode='a', header=False, index=False)
    if type(wpb_channel) == int:
        wpb_image = tiffStack[wpb_channel-1]
        wpb_blobs, wpb_labels, wpb_contours, wpb_boundaries, wpb_features = weibelPaladeSegmentation(wpb_image)
        print "    {0} wpb ".format(len(wpb_features))
        if single_cell_analysis == False: wpb_features.to_csv('../03_python_results/'+experiment_name+'_wpb.csv', sep=',', mode='a', header=False, index=False)
    if type(golgi_channel) == int:
        golgi_image = tiffStack[golgi_channel-1]
        golgi_blobs, golgi_labels, golgi_boundaries, golgi_features = golgiSegmentation(golgi_image)
        print "    {0} golgi fragments ".format(len(golgi_features))
        if single_cell_analysis == False: golgi_features.to_csv('../03_python_results/'+experiment_name+'_golgi.csv', sep=',', mode='a', header=False, index=False)
    if type(exitSites_channel) == int:
        exitSites_image = tiffStack[exitSites_channel-1]
        exit_sites_blobs, exit_sites_labels, exit_sites_boundaries, exit_sites_features = exitSitesSegmentation(exitSites_image, exit_sites_threshold, exit_sites_min_area)
        print "    {0} exit sites ".format(len(exit_sites_features))
        if single_cell_analysis == False: exit_sites_features.to_csv('../03_python_results/'+experiment_name+'_exit_sites.csv', sep=',', mode='a', header=False, index=False)

    if type(nucleus_channel) == int and single_cell_analysis == True:
        nucleus_features = assignCell(nucleus_labels, cell_labels, nucleus_features)
        nucleus_features.to_csv('../03_python_results/'+experiment_name+'_nuclei.csv', sep=',', mode='a', header=False, index=False)
    if type(plasmaMembrane_channel) == int and single_cell_analysis == True:
        cell_features = assignCell(cell_labels, cell_labels, cell_features)
        cell_features.to_csv('../03_python_results/'+experiment_name+'_cells.csv', sep=',', mode='a', header=False, index=False)
    if type(wpb_channel) == int and single_cell_analysis == True:
        wpb_features = assignCell(wpb_labels, cell_labels, wpb_features)
        wpb_features.to_csv('../03_python_results/'+experiment_name+'_wpb.csv', sep=',', mode='a', header=False, index=False)
    if type(golgi_channel) == int and single_cell_analysis == True:
        golgi_features = assignCell(golgi_labels, cell_labels, golgi_features)
        golgi_features.to_csv('../03_python_results/'+experiment_name+'_golgi.csv', sep=',', mode='a', header=False, index=False)
    if type(exitSites_channel) == int and single_cell_analysis == True:
        exit_sites_features = assignCell(exit_sites_labels, cell_labels, exit_sites_features)
        exit_sites_features.to_csv('../03_python_results/'+experiment_name+'_exit_sites.csv', sep=',', mode='a', header=False, index=False)

    #contours = np.add(nucleus_contours, cell_contours, wpb_contours)
    #composite_image = create_composite(nucleus, plasmaMembrane, wpb, contours)
    #imsave('../03_python_results/rgb_overlay/'+image_name.split('.',1)[0]+'.png', composite_image)

    #syntheticCoordinates(cell_labels, wpb_features)

print '\nAnalysis time: ', datetime.now() - start_time, ' seconds'
