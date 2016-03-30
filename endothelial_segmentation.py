import numpy as np
import pandas as pd
import skimage
import re
import os

from datetime import datetime
from os import listdir
from os.path import isfile, join
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, squareform
from skimage import measure, morphology, img_as_uint, img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
from skimage.filters import *
from skimage.io import imsave
from skimage.morphology import disk, watershed, remove_small_objects
from skimage.segmentation import *

start_time = datetime.now()

file_path = '../../01_TIF/'

pixel_dimension = 0.1615 #micro meter

nucleus_median_radius = 2 #pixels
nucleus_markers_sensitivity = 20
nucleus_min_area = 2000 #pixels

exit_sites_threshold = 796 #16-bit
exit_sites_min_area = 5 #pixels

wpb_block_size = 21 #adaptive threshold block size
wpb_offset = -1 #adaptive threshold offset
wpb_min_area = 5 #pixels
wpb_max_area = 385 #pixels

golgi_block_size = 20 #adaptive threshold block size
golgi_offset = -10 #adaptive threshold offset
golgi_min_area = 5 #pixels
golgi_max_area = 2000 #pixels

regex=re.compile("")
image_list = set([f for f in listdir(file_path) if isfile(join(file_path,f))])
exclude_list = set([m.group(0) for l in image_list for m in [regex.search(l)] if m])
image_list = sorted(image_list - exclude_list)#[51:56]
number_of_images = len(image_list)
experiment_name  = os.path.basename(os.path.dirname(os.path.dirname(os.getcwd())))

def scale8bit(image):
    scale = float(256) / (image.max() - image.min())
    return np.clip(np.round(np.multiply(image, scale)), 0, 255).astype(np.uint8)

def calculateFeret(coordinates):
    feret = np.nanmax(squareform(pdist(coordinates)))
    feret = feret + (((2*((0.5)**2))**(0.5))*2)
    return feret

def nucleusSegmentation(nucleus, nucleus_median_radius, nucleus_min_area, nucleus_markers_sensitivity):
    nucleus_rescale = rescale_intensity(nucleus, in_range=(0,nucleus.max()))
    nucleus_median = median(nucleus, disk(nucleus_median_radius))
    otsu_thresh = threshold_otsu(nucleus_median)
    binary = nucleus > otsu_thresh
    blobs = remove_small_objects(~(remove_small_objects(~binary, min_size=nucleus_min_area)), min_size=nucleus_min_area)
    distance = ndi.distance_transform_edt(blobs)
    blobs_labels = watershed(-distance, ndi.label(distance > nucleus_markers_sensitivity)[0], mask=blobs)
    blobs = blobs^find_boundaries(blobs_labels, mode='inner')
    blobs = remove_small_objects(~(remove_small_objects(~blobs, min_size=nucleus_min_area)), min_size=nucleus_min_area)
    blobs_labels = ndi.label(blobs)[0]
    boundaries = mark_boundaries(nucleus_rescale, blobs_labels, color=(0.6,0,0), outline_color=None, mode='outer')
    return (blobs, blobs_labels, boundaries)

def golgiSegmentation(golgi):
    binary = threshold_adaptive(golgi, golgi_block_size, offset=golgi_offset)
    blobs = remove_small_objects(binary, min_size=wpb_min_area) - remove_small_objects(binary, min_size=wpb_max_area)
    blobs_labels = ndi.label(blobs)[0]
    boundaries = mark_boundaries(scale8bit(golgi), blobs_labels, color=(0.6,0,0), outline_color=None, mode='outer')
    return (blobs, blobs_labels, boundaries)

def weibelPaladeSegmentation(wpb):
    wpb = img_as_ubyte(wpb)
    binary = threshold_adaptive(wpb, wpb_block_size, offset=wpb_offset)
    blobs = remove_small_objects(binary, min_size=wpb_min_area) - remove_small_objects(binary, min_size=wpb_max_area)
    blobs_labels = ndi.label(blobs)[0]
    boundaries = mark_boundaries(scale8bit(wpb), blobs_labels, color=(0.6,0,0), outline_color=None, mode='outer')
    return (blobs, blobs_labels, boundaries)

def exitSitesSegmentation(exitSites, threshold_value, exit_sites_min_area):
    blur = skimage.img_as_int(skimage.filters.gaussian_filter(exitSites, sigma=1))
    threshold = blur; threshold[threshold < threshold_value] = 0
    binary = threshold > exit_sites_threshold
    blobs = remove_small_objects(~(remove_small_objects(~binary, min_size=exit_sites_min_area)), min_size=exit_sites_min_area)
    local_maxi = peak_local_max(blur, min_distance=5, footprint=np.ones((9, 9)), labels=blobs, exclude_border=True, indices=False)
    blobs_labels = watershed(-blur, ndi.label(local_maxi)[0], mask=blobs)
    blobs_labels = morphology.remove_small_objects(blobs_labels, min_size=exit_sites_min_area)
    boundaries = mark_boundaries(scale8bit(exitSites), blobs_labels, color=(0.6,0,0), outline_color=None, mode='outer')
    return (blobs, blobs_labels, boundaries)

def cellSegmentation(pm, nucleus_blobs):
    pm = skimage.exposure.equalize_adapthist(pm, ntiles_x=20, clip_limit=0.01)
    pm_denoise = skimage.restoration.denoise_tv_chambolle(pm, weight=0.1)
    pm_seeds = np.fmin(~img_as_uint(nucleus_blobs), img_as_uint(pm_denoise))
    nucleus_markers = morphology.dilation((ndi.label(nucleus_blobs)[0]), disk(20))
    blobs_labels = watershed(pm_seeds, nucleus_markers)
    blobs = clear_border(~np.zeros(blobs_labels.shape, dtype=bool)^find_boundaries(blobs_labels, mode='inner'))
    blobs_labels = ndi.label(blobs)[0]
    boundaries = mark_boundaries(pm, blobs_labels, color=(0.6,0,0), outline_color=None, mode='thick')
    return (blobs, blobs_labels, boundaries)

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

cols = ['particle_id','row','col','fov','cell','x_centroid', 'y_centroid','area','perimeter','feret','equivalent_diameter','convex_area','major_axis_length','minor_axis_length','orientation','solidity','max_intensity','min_intensity','mean_intensity']
pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_nuclei.csv', sep=',', header=True, index=False)
pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_cells.csv', sep=',', header=True, index=False)
pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_wpb.csv', sep=',', header=True, index=False)
#pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_golgi.csv', sep=',', header=True, index=False)
#pd.DataFrame(columns = cols).to_csv('../03_python_results/'+experiment_name+'_exit_sites.csv', sep=',', header=True, index=False)

#pd.DataFrame(columns = cols+['x_synth','y_synth']).to_csv('../03_python_results/'+experiment_name+'_wpb.csv', sep=',', header=True, index=False)

for image in range(0, number_of_images):
    image_name = image_list[image]
    tiffStack = skimage.io.imread(file_path+image_name, plugin='tifffile')

    nucleus = tiffStack[0]
    wpb = tiffStack[1]
    #golgi = tiffStack[2]
    #exitSites = tiffStack[1]
    plasmaMembrane = tiffStack[2]

    nucleus_blobs, nucleus_labels, nucleus_boundaries = nucleusSegmentation(nucleus, nucleus_median_radius, nucleus_min_area, nucleus_markers_sensitivity)
    cell_blobs, cell_labels, cell_boundaries = cellSegmentation(plasmaMembrane, nucleus_blobs)
    wpb_blobs, wpb_labels, wpb_boundaries = weibelPaladeSegmentation(wpb)
    #golgi_blobs, golgi_labels, golgi_boundaries = golgiSegmentation(golgi)
    #exit_sites_blobs, exit_sites_labels, exit_sites_boundaries = exitSitesSegmentation(exitSites, exit_sites_threshold, exit_sites_min_area)

    nucleus_features = measureMorphometry(nucleus_labels, nucleus, image_name)
    cell_features = measureMorphometry(cell_labels, plasmaMembrane, image_name)
    wpb_features = measureMorphometry(wpb_labels, wpb, image_name)
    #golgi_features = measureMorphometry(golgi_labels, golgi, image_name)
    #exit_sites_features = measureMorphometry(exit_sites_labels, exitSites, image_name)

    nucleus_features = assignCell(nucleus_labels, cell_labels, nucleus_features)
    wpb_features = assignCell(wpb_labels, cell_labels, wpb_features)
    #exit_sites_features = assignCell(exit_sites_labels, cell_labels, exit_sites_features)
    cell_features = assignCell(cell_labels, cell_labels, cell_features)

    #syntheticCoordinates(cell_labels, wpb_features)

    imsave('../03_python_results/cell_labels/'+image_name.split('.',1)[0]+'.png', cell_labels)

    imsave('../03_python_results/nucleus_overlay/'+image_name.split('.',1)[0]+'.png', nucleus_boundaries)
    imsave('../03_python_results/cell_overlay/'+image_name.split('.',1)[0]+'.png', cell_boundaries)
    imsave('../03_python_results/wpb_overlay/'+image_name.split('.',1)[0]+'.png', wpb_boundaries)
    #imsave('../03_python_results/golgi_overlay/'+image_name.split('.',1)[0]+'.png', golgi_boundaries)
    #imsave('../03_python_results/exit_sites_overlay/'+image_name.split('.',1)[0]+'.png', exit_sites_boundaries)

    nucleus_features.to_csv('../03_python_results/'+experiment_name+'_nuclei.csv', sep=',', mode='a', header=False, index=False)
    cell_features.to_csv('../03_python_results/'+experiment_name+'_cells.csv', sep=',', mode='a', header=False, index=False)
    wpb_features.to_csv('../03_python_results/'+experiment_name+'_wpb.csv', sep=',', mode='a', header=False, index=False)
    #wpb_features.to_csv('../03_python_results/'+experiment_name+'_golgi.csv', sep=',', mode='a', header=False, index=False)
    #exit_sites_features.to_csv('../03_python_results/'+experiment_name+'_exit_sites.csv', sep=',', mode='a', header=False, index=False)

    print "Analysing {0}, image {1} of {2}, detected {3} wpb and {4} nuclei".format(image_name, image+1, number_of_images, len(wpb_features), len(nucleus_features))

print '\nAnalysis time: ', datetime.now() - start_time, ' seconds'
