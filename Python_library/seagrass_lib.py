#####################
# Import Libraries
#####################
import zipfile
import fnmatch
import os
from xml.dom import minidom
import pandas as pd
import datetime
import numpy as np
import gc 
import glob
import tensorflow
from keras import __version__
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Convolution2D
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.saving import hdf5_format
import sys
import h5py
from osgeo import gdal, osr, ogr
import subprocess
import shutil
from sklearn.metrics import confusion_matrix
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import spectral
import rasterio as rio
from rasterio import plot as rasterplot
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.enums import Resampling
import rioxarray
from shapely.geometry import mapping
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import fiona
import seaborn as sns


#####################
# Supporting Functions
##################### Create a specified directory if it doesn't exist
def create_directory(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)
##################### Recursive file search
def recursive_search(input_dir, pattern = "*.TIF"): # By default the function will recursively search for files with .TIF extension
    file_list = [] # Empty list to store files
    for root, dirnames, filenames in os.walk(input_dir):
        for filename in fnmatch.filter(filenames, pattern):
            file_list.append(os.path.join(root, filename))
    return(file_list) # Return list of files with specified extension
##################### Extract a specified instance of a metadata variable from a parsed .XML file
def extract_metadata(metadata_file, metadata_var, instance = 0):
    item = metadata_file.getElementsByTagName(metadata_var)[instance]
    val = ''.join([node.data for node in item.childNodes])
    return(val) # Return value as a string
##################### Basic check to determine if the correct number of coordinates have been supplied to define an extent
def check_extent_coord(ul_lon, ul_lat, lr_lon, lr_lat):
    return(None) # Don't return anything
##################### Convert upper left and lower right lat/lon coordinates to a polygon
def coord_to_extent_polygon(extent_coord):
    check_extent_coord(*extent_coord) # Check if the appropriate number of coordinates have been supplied
    ring = ogr.Geometry(ogr.wkbLinearRing) # Create linear ring geometry
    ring.AddPoint(extent_coord[0], extent_coord[1]) # Add extent coordinates as points to linear string
    ring.AddPoint(extent_coord[2], extent_coord[1]) # ...
    ring.AddPoint(extent_coord[2], extent_coord[3]) # ...
    ring.AddPoint(extent_coord[0], extent_coord[3]) # ...
    ring.AddPoint(extent_coord[0], extent_coord[1]) # Close linear string 
    poly = ogr.Geometry(ogr.wkbPolygon) # Create polygon geometry
    poly.AddGeometry(ring) # Add linear string to polygon
    return(poly) # Return the polygon
##################### Convert lat, lon to row, column for an image that uses rational polynomical coefficients (RPCs) for spatial reference
def coord_to_rpc_image(lon, lat, rpc_coeff, height = 0):
    L = (lon - float(rpc_coeff.get('LONG_OFF'))) / float(rpc_coeff.get('LONG_SCALE'))
    P = (lat - float(rpc_coeff.get('LAT_OFF'))) / float(rpc_coeff.get('LAT_SCALE'))
    H = (height - float(rpc_coeff.get('HEIGHT_OFF'))) / float(rpc_coeff.get('HEIGHT_SCALE'))
    # convert each set of RPCss to a list of floats
    samp_den_coeff = [float(coeff) for coeff in rpc_coeff.get('SAMP_DEN_COEFF').split()]
    samp_num_coeff = [float(coeff) for coeff in rpc_coeff.get('SAMP_NUM_COEFF').split()]
    line_den_coeff = [float(coeff) for coeff in rpc_coeff.get('LINE_DEN_COEFF').split()]
    line_num_coeff = [float(coeff) for coeff in rpc_coeff.get('LINE_NUM_COEFF').split()]
    # calculate the density and number values for line (row) and sample (col) using the RPCs
    samp_den = samp_den_coeff[0] + (samp_den_coeff[1] * L) + (samp_den_coeff[2] * P) + (samp_den_coeff[3] * H) + (samp_den_coeff[4] * L * P) + (samp_den_coeff[5] * L * H) + (samp_den_coeff[6] * P * H) + (samp_den_coeff[7] * L**2) + (samp_den_coeff[8] * P**2) + (samp_den_coeff[9] * H**2) + (samp_den_coeff[10] * L * P * H) + (samp_den_coeff[11] * L**3) + (samp_den_coeff[12] * L * P**2) + (samp_den_coeff[13] * L * H**2) + (samp_den_coeff[14] * L**2 * P) + (samp_den_coeff[15] * P**3) + (samp_den_coeff[16] * P * H**2) + (samp_den_coeff[17] * L**2 * H) + (samp_den_coeff[18] * P**2 * H) + (samp_den_coeff[19] * H**3)
    samp_num = samp_num_coeff[0] + (samp_num_coeff[1] * L) + (samp_num_coeff[2] * P) + (samp_num_coeff[3] * H) + (samp_num_coeff[4] * L * P) + (samp_num_coeff[5] * L * H) + (samp_num_coeff[6] * P * H) + (samp_num_coeff[7] * L**2) + (samp_num_coeff[8] * P**2) + (samp_num_coeff[9] * H**2) + (samp_num_coeff[10] * L * P * H) + (samp_num_coeff[11] * L**3) + (samp_num_coeff[12] * L * P**2) + (samp_num_coeff[13] * L * H**2) + (samp_num_coeff[14] * L**2 * P) + (samp_num_coeff[15] * P**3) + (samp_num_coeff[16] * P * H**2) + (samp_num_coeff[17] * L**2 * H) + (samp_num_coeff[18] * P**2 * H) + (samp_num_coeff[19] * H**3)
    line_den = line_den_coeff[0] + (line_den_coeff[1] * L) + (line_den_coeff[2] * P) + (line_den_coeff[3] * H) + (line_den_coeff[4] * L * P) + (line_den_coeff[5] * L * H) + (line_den_coeff[6] * P * H) + (line_den_coeff[7] * L**2) + (line_den_coeff[8] * P**2) + (line_den_coeff[9] * H**2) + (line_den_coeff[10] * L * P * H) + (line_den_coeff[11] * L**3) + (line_den_coeff[12] * L * P**2) + (line_den_coeff[13] * L * H**2) + (line_den_coeff[14] * L**2 * P) + (line_den_coeff[15] * P**3) + (line_den_coeff[16] * P * H**2) + (line_den_coeff[17] * L**2 * H) + (line_den_coeff[18] * P**2 * H) + (line_den_coeff[19] * H**3)
    line_num = line_num_coeff[0] + (line_num_coeff[1] * L) + (line_num_coeff[2] * P) + (line_num_coeff[3] * H) + (line_num_coeff[4] * L * P) + (line_num_coeff[5] * L * H) + (line_num_coeff[6] * P * H) + (line_num_coeff[7] * L**2) + (line_num_coeff[8] * P**2) + (line_num_coeff[9] * H**2) + (line_num_coeff[10] * L * P * H) + (line_num_coeff[11] * L**3) + (line_num_coeff[12] * L * P**2) + (line_num_coeff[13] * L * H**2) + (line_num_coeff[14] * L**2 * P) + (line_num_coeff[15] * P**3) + (line_num_coeff[16] * P * H**2) + (line_num_coeff[17] * L**2 * H) + (line_num_coeff[18] * P**2 * H) + (line_num_coeff[19] * H**3)
    # use the number and density values to calculate the row and column position
    c_n = samp_num / samp_den
    col = int((c_n * float(rpc_coeff.get('SAMP_SCALE'))) + float(rpc_coeff.get('SAMP_OFF')))
    r_n = line_num / line_den
    row = int((r_n * float(rpc_coeff.get('LINE_SCALE'))) + float(rpc_coeff.get('LINE_OFF')))
    return([col,row]) # return column and row position
##################### Convert lat, lon to x,y coordinates in a projected image's coordinate system
def coord_to_proj_image(lon, lat, projection):
    target = osr.SpatialReference(wkt = projection) # Target spatial reference object
    source = osr.SpatialReference() # Empty spatial reference object
    source.ImportFromEPSG(4326) # Set the source spatial reference object to EPSG:4326 or lat,lon
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) # Force the source spatial reference object to except the traditional lon, lat instead of lat, lon
    transform = osr.CoordinateTransformation(source, target) # Create spatial reference transformation object
    point = ogr.Geometry(ogr.wkbPoint) # Create point geometry
    point.AddPoint(lon, lat) # Add coordinates as a point
    point.Transform(transform) # Transform the point object from the source to target spatial reference
    return([point.GetX(), point.GetY()]) # Return x, y
##################### Convert x,y coordinates in a projected image's coordinate system to row, column
def world_to_pixel(x, y, geotransform):
    ul_x = geotransform[0] # Upper left x coordinate
    ul_y = geotransform[3] # Upper left y coordinate
    x_dist = geotransform[1] # Pixel spacing in the x direction
    y_dist = geotransform[5] # Pixel spacing in the y direction
    col = int((x - ul_x) / x_dist) # Column position
    row = -int((ul_y - y) / y_dist) # Row position
    return([col, row]) # Return col and row position
#####################
def mask_extent(ds, aoi_extent):
    check_extent_coord(*aoi_extent) # Create output directory if it doesn't already exist
    ul_lon, ul_lat, lr_lon, lr_lat = aoi_extent # Split list of extent coordinates into individual objects
    if(len(ds.GetMetadata('RPC')) > 0): # If the input image is using rational polynomial coefficients for spatial reference
        ul_col, ul_row = coord_to_rpc_image(lat = ul_lat, lon = ul_lon, height = 0, rpc_coeff = ds.GetMetadata('RPC'))
        lr_col, lr_row = coord_to_rpc_image(lat = lr_lat, lon = lr_lon, height = 0, rpc_coeff = ds.GetMetadata('RPC'))
    else: # If the input image is using a coordinate system 
        ul_x, ul_y = coord_to_proj_image(lon = ul_lon, lat = ul_lat, projection = ds.GetProjection()) # Convert the upper left lat,lon coordinates to x,y coordinates in the image coordinate system
        lr_x, lr_y = coord_to_proj_image(lon = lr_lon, lat = lr_lat, projection = ds.GetProjection()) # Convert the lower right lat,lon coordinates to x,y coordinates in the image coordinate system
        ul_col, ul_row = world_to_pixel(x = ul_x, y = ul_y, geotransform = ds.GetGeoTransform()) # Convert the upper left x,y coordinates to col,row position within the input image
        lr_col, lr_row = world_to_pixel(x = lr_x, y = lr_y, geotransform = ds.GetGeoTransform()) # Convert the lower right x,y coordinates to col,row position within the input image
    if(ul_col < 0): # If the specified ul_lon is west of the image, set ul_lat to the origin 
        ul_col = 0
    if(ul_row < 0): # If the specified ul_lat is north of the image, set ul_lat to the origin 
        ul_row = 0
    if(lr_col > (ds.RasterXSize - 1)): # If the specified lr_lon is east of the image, set lr_lon to the max col position
        lr_col = ds.RasterXSize - 1
    if(lr_row > (ds.RasterYSize - 1)): # If the specified lr_lat is south of the image, set lr_lat to the max row position
        lr_row = ds.RasterYSize - 1
    if((ul_col in range(0,ds.RasterXSize)) & (lr_col in range(0,ds.RasterXSize)) & (ul_row in range(0,ds.RasterYSize)) & (lr_row in range(0,ds.RasterYSize))): # If the ul & lr coordinates overlap the image at all return the col & row positions
        return([ul_col, ul_row, lr_col, lr_row])
    else: # Otherwise return none
        return([None, None, None, None])
##################### Parse a worldview date-time string
def parse_date_time(date_time_string):
    parsed = datetime.datetime.strptime(date_time_string.split('T')[0], '%Y-%m-%d').timetuple()
    return(parsed) # Return a tuple containing the parsed, consituent components of the input date-time object
##################### Take a parsed worldview date-time object and return the sun-earth distance in astronomical units
def earth_sun_distance(parsed_date_time):
    yr = parsed_date_time.tm_year # Year
    mon = parsed_date_time.tm_mon # Month
    d = parsed_date_time.tm_mday # Day
    if mon in [1,2]: # If the image was acquired in January or February, modify the year and month
        yr = yr - 1
        mon = mon + 12
    UT = parsed_date_time.tm_hour + (parsed_date_time.tm_min/60.0) + (parsed_date_time.tm_sec/3600.0) # Universal Time
    A = int(yr/100)
    B = 2 - A + int(A/4)
    JD = int(365.25*(yr + 4716)) + int(30.6001*(mon + 1)) + d + (UT/24.0) + B - 1524.5 # Julian Day
    D = JD - 2451545.0 
    g = 357.529 + 0.98560028 * D
    d_ES = 1.00014 - 0.01671 * np.cos(g*(np.pi/180)) - 0.00014 * np.cos(2*g*(np.pi/180)) # The Earth-Sun distance in Astronomical Units (AU). Should have a value between 0.983 and 1.017
    return(d_ES) # Return the Earth-Sun distance in AU
##################### Gain values for WV2 & WV3 sensors
def worldview_gain(sat_id, band_num):
    if sat_id == 'WV02':
        return([1.151, 0.988, 0.936, 0.949, 0.952, 0.974, 0.961, 1.002][band_num])
    elif sat_id == 'WV03':
        return([0.905, 0.940, 0.938, 0.962, 0.964, 1.000, 0.961, 0.978][band_num])
##################### Offset values for WV2 & WV3 sensors
def worldview_offset(sat_id, band_num):
    if sat_id == 'WV02':
        return([-7.478, -5.736, -3.546, -3.564, -2.512, -4.120, -3.300, -2.891][band_num])
    elif sat_id == 'WV03':
        return([-8.604, -5.809, -4.996, -3.649, -3.021, -4.521, -5.522, -2.992][band_num])
##################### Band averaged extraterrestrial solar irridiance for WV2 & WV3 sensors
def worldview_eai(sat_id, band_num):
    if sat_id == 'WV02':
        return([1773.81, 2007.27, 1829.62, 1701.85, 1538.85, 1346.09, 1053.21, 856.599][band_num])
    elif sat_id == 'WV03':
        return([1757.89, 2004.61, 1830.18, 1712.07, 1535.33, 1348.08, 1055.94, 858.77][band_num])
##################### Band centers for WV2 & WV3 sensors
def worldview_band_center(sat_id, band_num):
    if sat_id == 'WV02':
        return([0.4273, 0.4779, 0.5462, 0.6078, 0.6588, 0.7237, 0.8313, 0.9080][band_num])
    elif sat_id == 'WV03':
        return([0.4274 , 0.4819, 0.5471, 0.6043, 0.6601, 0.7227, 0.8240, 0.9136][band_num]) 
#####################
# Main Functions
##################### extract info about the filepaths, cloud cover, and aoi coverage of the multispectral tiles packaged in a zipped folder delivered by Maxar
def list_files(zip_fp, aoi_extent):
    check_extent_coord(*aoi_extent) # Check if the appropriate number of coordinates have been supplied
    aoi_polygon = coord_to_extent_polygon(aoi_extent) # Create polygon object out of AOI extent coordinates
    f_zip = zipfile.ZipFile(zip_fp, 'r') # Read zipped folder
    f_list = f_zip.namelist() # List of the files in the zipped folder
    f_xml = fnmatch.filter(f_list, "*MUL*.XML") # Filter the list of files contained in the zipped folder to extract the XML files for the multispectral tiles
    aquisitionDF = pd.DataFrame() # Create dataframe to store image information
    for i in range(0, len(f_xml)): # Loop through each XML file
        f = f_zip.open(f_xml[i]) # Open ith XML file
        parsed_metadata = minidom.parse(f) # Parse XML file
        aquisitionDF.at[i,'DIRECTORY'] = os.path.dirname(f_xml[i]) # Save directory name for the XML file (this directory also contains the image data that we ultimately want)
        for var in ['SATID', 'TLCTIME','CLOUDCOVER']: # Extract the satellite ID, acquisition date-time, and cloud cover from the parsed XML file
            aquisitionDF.at[i,var] = extract_metadata(metadata_file = parsed_metadata, metadata_var = var, instance = 0) 
        coord_list = ['ULLON', 'ULLAT', 'URLON','URLAT', 'LRLON', 'LRLAT', 'LLLON','LLLAT', 'ULLON', 'ULLAT'] # List the variable names for the corner coordinates of the image footprint
        geom = [] # Empty list to store coordinates
        for coord in coord_list: # Extract corner coordinates for image footprint from metadata
            geom.append(float(extract_metadata(metadata_file = parsed_metadata, metadata_var = coord, instance = 0)))
        footprint_ring = ogr.Geometry(ogr.wkbLinearRing) # Create linear ring geometry
        for lon,lat in zip(geom[0::2], geom[1::2]): # Add corner coordinates as points along linear ring
            footprint_ring.AddPoint(lon,lat)
        footprint_polygon = ogr.Geometry(ogr.wkbPolygon) # Create polygon geometry
        footprint_polygon.AddGeometry(footprint_ring) # Convert linear ring to polygon
        intersection_polygon = footprint_polygon.Intersection(aoi_polygon) # Get intersection of image footprint and AOI extent
        percent_overlap = (intersection_polygon.GetArea() / footprint_polygon.GetArea()) * 100 # Compute the percent of the image that overlaps the AOI
        aquisitionDF.at[i,'AOI_COVERAGE'] = round(percent_overlap, 2) # Store the percentage overlap, rounded to the 100th
    return(aquisitionDF) # Return dataframe containing information about the multipsectral tiles within the zipped folder
##################### unzip specified files
def unzip_tiles(zip_fp, tile_dir, output_dir):
    create_directory(output_dir) # Create output directory if it doesn't already exist
    f_zip = zipfile.ZipFile(zip_fp, 'r') # Read zipped folder
    f_list = f_zip.namelist() # List files within zipped folder
    for directory in tile_dir: # Loop through the list of tile directories that should be unzipped
        f_match = fnmatch.filter(f_list, directory + "*") # List all files within tile directory
        for f in f_match: # Extract each file within tile directory
            f_zip.extract(f, output_dir)
##################### clip image by aoi extent
def clip_image(image_fp, output_fp, aoi_extent, ds_nodata = None):
    create_directory(os.path.dirname(output_fp)) # Create an output directory if it doesn't already exist
    check_extent_coord(*aoi_extent) # Check if the appropriate number of coordinates were given
    ul_lon, ul_lat, lr_lon, lr_lat = aoi_extent # Split list of AOI extent coordinates into individual objects
    ds = gdal.Open(image_fp) # Open input image
    if(ds_nodata == None):
        ds_nodata = ds.GetRasterBand(1).GetNoDataValue()
    ul_x, ul_y = coord_to_proj_image(lon = ul_lon, lat = ul_lat, projection = ds.GetProjection()) # Convert the upper left lat,lon coordinates to x,y coordinates in the image coordinate system
    lr_x, lr_y = coord_to_proj_image(lon = lr_lon, lat = lr_lat, projection = ds.GetProjection()) # Convert the lower right lat,lon coordinates to x,y coordinates in the image coordinate system
    ul_col, ul_row = world_to_pixel(x = ul_x, y = ul_y, geotransform = ds.GetGeoTransform()) # Convert the upper left x,y coordinates to col,row position within the input image
    lr_col, lr_row = world_to_pixel(x = lr_x, y = lr_y, geotransform = ds.GetGeoTransform()) # Convert the lower right x,y coordinates to col,row position within the input image
    if(ul_col < 0): # if the specified ul_lon is west of the image, set ul_lat to the origin 
        ul_col = 0
    if(ul_row < 0): # if the specified ul_lat is north of the image, set ul_lat to the origin 
        ul_row = 0
    if(lr_col > (ds.RasterXSize - 1)): # if the specified lr_lon is east of the image, set lr_lon to the max col position
        lr_col = ds.RasterXSize - 1
    if(lr_row > (ds.RasterYSize - 1)): # if the specified lr_lat is south of the image, set lr_lat to the max row position
        lr_row = ds.RasterYSize - 1
    cmd = ["gdal_translate", "-srcwin" , str(ul_col), str(ul_row), str(lr_col - ul_col), str(lr_row - ul_row), "-a_nodata", str(ds_nodata), "-eco", image_fp, output_fp] # Clip command
    subprocess.call(cmd) # Run clip command on the command line
    ds.FlushCache() # Flush image cache to avoid read/write errors
    ds = None
    if(os.path.exists(output_fp) & os.path.exists(image_fp.replace(".TIF", ".XML"))): # Copy image metadata from the input image directory to output image directory if it exists
        output_dir = os.path.dirname(output_fp)
        new_xml = os.path.basename(image_fp).replace(".TIF", ".XML")
        shutil.copyfile(image_fp.replace(".TIF", ".XML"), os.path.join(output_dir, new_xml))
##################### project image
def project_image(image_fp, output_fp, target_coord, res, rpc = False, resampling_method = "bilinear"):
    create_directory(os.path.dirname(output_fp)) # Create output directory if it doesn't already exist
    if(rpc == False): # Create projection command for an image that isn't using rational polynomial coefficients
        cmd = ["gdalwarp", "-t_srs", target_coord, "-tr", str(res), str(res), "-r", resampling_method, image_fp, output_fp]
    elif(rpc == True): # Create projection command for an image that is using rational polynomial coefficients
        cmd = ["gdalwarp", "-rpc", "-t_srs" , target_coord, "-tr", str(res), str(res), "-r", resampling_method, image_fp, output_fp]
    subprocess.call(cmd) # Run projection command on the command line
    if(os.path.exists(output_fp) & os.path.exists(image_fp.replace(".TIF", ".XML"))): # If an .XML metadata file exists in the same directory as the input image, copy it to the output directory
        output_dir = os.path.dirname(output_fp)
        new_xml = os.path.basename(image_fp).replace(".TIF", ".XML")
        shutil.copyfile(image_fp.replace(".TIF", ".XML"), os.path.join(output_dir, new_xml))
##################### Convert Level 1B image to remote sensing reflectance
def rad_cal(image_fp, output_fp, aoi_extent = None, dst_nodata = -32768):
    create_directory(os.path.dirname(output_fp)) # Create output directory if it doesn't already exist
    ds = gdal.Open(image_fp) # Open input image
    ds_geotransform = ds.GetGeoTransform() # Get image geotransform
    ds_projection = ds.GetProjection() # Get image projection. If the image is using rational polynomial coefficients for spatial reference, this will be blank 
    ds_RPCs = ds.GetMetadata('RPC') # Get image rational polynomial coefficients. If the image is projected, this will be blank 
    ds_nbands = ds.RasterCount # Get the number of bands
    ds_cols = ds.RasterXSize # Get the number of columns (x)
    ds_rows = ds.RasterYSize # Get the number of rows (y)
    ul_x, ul_y, lr_x, lr_y = [0, 0, ds_cols, ds_rows] # Establish default row, col processing extent (entire image)
    if(aoi_extent != None): # If an AOI extent was provided as an argument find the corresponding row, col positions and update the processing extent
        mask_pos = mask_extent(ds, aoi_extent)
        if(None in mask_pos):
            print("The specified extent does not overlap the image footprint. Proceeding with the whole image.")
        else:
            ul_x, ul_y, lr_x, lr_y = mask_pos
    metadata_fp = image_fp.replace(".TIF", ".XML") # Input image metadata filepath (should be in the same directory as input image)
    metadata = minidom.parse(metadata_fp) # Parse metadata .XML file
    solar_elevation = extract_metadata(metadata_file = metadata, metadata_var = 'MEANSUNEL', instance = 0) # Get solar elevation from the metadata (in degrees)
    solar_zenith = 90 - float(solar_elevation) # Convert solar elevation to solar zenith (in degrees)
    aquisition_date_time = extract_metadata(metadata_file = metadata, metadata_var = 'TLCTIME', instance = 0) # Get aquisition date-time object from the metadata
    parsed = parse_date_time(aquisition_date_time) # Parse data-time object into its constituent components
    dist_au = earth_sun_distance(parsed) # Convert parsed date-time object to sun-earth distance in astronomical units  
    satellite = extract_metadata(metadata_file = metadata, metadata_var = 'SATID', instance = 0) # Get the satellite id from the metadata (WV2 or WV3)
    
    dst_ds = gdal.GetDriverByName('GTiff').Create(output_fp, ds_cols, ds_rows, ds_nbands, gdal.GDT_Int16) # Create an empty raster that will contain radiometrically calibrated (remote sensing reflectance) pixel values
    dst_ds.SetGeoTransform(ds_geotransform) # Use the same geotranform as the input image
    dst_ds.SetProjection(ds_projection) # Use the same projection as the input image
    dst_ds.SetMetadata(ds_RPCs, 'RPC') # Use the same RPCs as the input image
    
    for j in range(ds_nbands): # Loop through each band of the input image
        ds_band = ds.GetRasterBand(j+1) # Get the jth band (python indexing starts at zero, gdal indexing starts at 1, hence j+1)
        ds_band_nodata = ds_band.GetNoDataValue() # Get the band's no data value
        col_bSize = ds_band.GetBlockSize()[0] # Get the TIF read/write block size (x/col dimension)
        row_bSize = ds_band.GetBlockSize()[1] # Get the TIF read/write block size (y/row dimension)
        abs_cal = float(extract_metadata(metadata_file = metadata, metadata_var = 'ABSCALFACTOR', instance = j)) # Get the absolute calibration factor from the metadata
        eff_band_width = float(extract_metadata(metadata_file = metadata, metadata_var = 'EFFECTIVEBANDWIDTH', instance = j)) # Get the effective bandwidth from the metadata
        gain = worldview_gain(sat_id = satellite, band_num = j) # Set the band gain value
        offset = worldview_offset(sat_id = satellite, band_num = j) # Set the band offset value
        solar_irradiance = worldview_eai(sat_id = satellite, band_num = j) # Set the band EAI value
        dst_ds.GetRasterBand(j+1).Fill(dst_nodata) # Fill output raster band with the specified no data value
        for cpos in range(ul_x, lr_x, col_bSize): # Loop through and process image in blocks
            if cpos + col_bSize < lr_x:
                numCols = col_bSize
            else:
                numCols = lr_x - cpos
            for rpos in range(ul_y, lr_y, row_bSize):
                if rpos + row_bSize < lr_y:
                    numRows = row_bSize
                else:
                    numRows = lr_y - rpos
                band_arr = ds_band.ReadAsArray(cpos, rpos, numCols, numRows) # Read image block
                mask = (band_arr != ds_band_nodata) # Create no data mask
                Rrs = np.full(np.shape(band_arr), float(dst_nodata)) # Create array object to store remote sensing reflectance pixel values
                if mask.any(): # Proceed if there are elements in the image block that aren't no data
                    rad_cal = np.full(np.shape(band_arr), float(dst_nodata)) # Create array object to store radiometrically calibrate pixel values
                    rad_cal[mask] = (band_arr[mask] * gain * (abs_cal/eff_band_width)) + offset # Convert Level 1B digital numbers to top of atmosphere radiance
                    Rrs[mask] = (((rad_cal[mask] * dist_au**2 * np.pi) / (solar_irradiance * np.cos(solar_zenith*(np.pi/180)))) / np.pi) * 10000 # Convert top of atmosphere radiance to remote sensing reflectance
                    dst_ds.GetRasterBand(j+1).WriteArray(Rrs, cpos, rpos) # Write array containing remote sensing reflectance to the output image object
        dst_ds.GetRasterBand(j+1).SetNoDataValue(dst_nodata) # Set no data value for the band
    dst_ds.SetMetadata({'sat_id': ''.join(satellite)}) # Embed satellite id into the image's metadata under the 'sat_id' tag
    # Get rid of unneeded objects to avoid read/write errors 
    dst_ds.FlushCache()
    ds.FlushCache()
    dst_ds = None
    ds = None
    rad_cal = None
    Rrs = None
    band_arr = None
    metadata = None
    gc.collect()
##################### calculate DOS value for an image and embed it the metadata of the image
def embed_dos_val(image_fp, green_band = 2, nir_band = 6, dos_band = 5, ndwi_threshold = 0, ds_nodata = None):
    ds = gdal.Open(image_fp, gdal.GA_Update) # Open input image
    cols = ds.RasterXSize # Number of columns in the input image
    rows = ds.RasterYSize  # Number of rows in the input image
    bSize = ds.GetRasterBand(1).GetBlockSize() # Input image block size
    if(ds_nodata == None): # If no data value wasn't supplied, get the no data value from the input image metadata
        ds_nodata = ds.GetRasterBand(1).GetNoDataValue()
    dos_arr = np.full([rows,cols], ds_nodata)  # Create an array with the same col, row dimensions as input image to store pixel values from the DOS band that overlap water areas
    for cpos in range(0, cols, bSize[0]): # Loop through blocks of the input image
        if cpos + bSize[0] < cols:
            numCols = bSize[0]
        else:
            numCols = cols - cpos
        for rpos in range(0, rows, bSize[1]):
            if rpos + bSize[1] < rows:
                numRows = bSize[1]
            else:
                numRows = rows - rpos
            green_block = ds.GetRasterBand(green_band + 1).ReadAsArray(cpos, rpos, numCols, numRows) # Get block of the green band
            mask = (green_block != ds_nodata) # Create array indicating where the no data values are within the block
            if mask.any(): # If the block contains any valid pixels
                ndwi = np.full(np.shape(green_block), ds_nodata) # Create an array to store ndwi values
                nir_block = ds.GetRasterBand(nir_band + 1).ReadAsArray(cpos, rpos, numCols, numRows) # Read block of the NIR1 band
                ndwi[mask] = np.divide(np.subtract(green_block[mask], nir_block[mask]), np.add(green_block[mask],nir_block[mask])) # Compute ndwi with the green block and NIR1 block
                pos = np.where(ndwi < ndwi_threshold) # Threshold ndwi values to find column and row positions where water isn't present 
                dos_block = ds.GetRasterBand(dos_band + 1).ReadAsArray(cpos, rpos, numCols, numRows) # Read block of the DOS band
                dos_block[pos[0], pos[1]] = ds_nodata # Mask pixels within the DOS band where water isn't present
                dos_arr[rpos:(rpos+numRows), cpos:(cpos+numCols)] = dos_block # Write DOS band block to the DOS band array
    dos_vec = dos_arr.flatten() # Flatten the DOS band array to 1D vector
    dos_vec = dos_vec[np.where(dos_vec != ds_nodata)] # Remove null values from 1D vector
    dos_sort = np.sort(dos_vec) # Sort 1D vector
    dos_val = np.median(dos_sort[0:round((len(dos_sort)/20)-1)])/2 # Find the DOS anchor value (half of the median of the lowest 5%) 
    satellite = ds.GetMetadata().get('sat_id') # Get the satellite id from the input image metadata
    if(satellite == None): # If the satellite id isn't present within the metadata, embed the DOS value into the input image metadata alone
        ds.SetMetadata({'dos_value': ''.join(str(dos_val))}) 
    elif(satellite != None): # If the satellite id is present within the metadata, embed the DOS value into the input image metadata along with the satellite id
        ds.SetMetadata({'sat_id': ''.join(satellite), 'dos_value': ''.join(str(dos_val))}) 
    ds.FlushCache() # Get rid of unneeded objects to avoid read/write errors
    ds = None
    ndwi = None
    green_block = None
    nir_block = None
    dos_block = None
    dos_arr = None
    dos_vec = None
    dos_sort = None
    pos = None
    gc.collect()
##################### Returns the minimum DOS value for a list of images
def min_dos_value(image_list):
    if isinstance(image_list, list): # Check if input object is a list
        dos_list = [] # Create object to store DOS values from each image
        for i in range(len(image_list)): # Loop through the list of images
            ds = gdal.Open(image_list[i]) # Open image
            tmp = float(ds.GetMetadata().get('dos_value')) # Get the DOS value embedded in the images metadata under the tag 'dos_value'
            dos_list.append(tmp) # Append DOS value to list of DOS values
            return(np.min(dos_list)) # Return the minimum DOS value as a float
    else:
        print("The input object is not a list. Try again")
##################### Perform DOS atmospheric correction
def atm_cor(image_fp, output_fp, rayleighExp, dos_band = 5, dos_value = None, satellite = None):
    create_directory(os.path.dirname(output_fp)) # create output directory if it doesn't exist
    ds = gdal.Open(image_fp) # Open input image
    ds_geotransform = ds.GetGeoTransform() # Get image geotransform
    ds_projection = ds.GetProjection() # Get image projection. If the image is using rational polynomial coefficients for spatial reference, this will be blank 
    ds_RPCs = ds.GetMetadata('RPC') # Get image rational polynomial coefficients. If the image is projected, this will be blank
    ds_nbands = ds.RasterCount # Get the number of bands
    ds_cols = ds.RasterXSize # Get the number of columns (x)
    ds_rows = ds.RasterYSize # Get the number of rows (y)
    ds_nodata = ds.GetRasterBand(1).GetNoDataValue()
    if(dos_value == None): # If DOS value wasn't supplied as an argument, use the DOS value that is embedded in the image's metadata
        dos_value = float(ds.GetMetadata().get('dos_value'))
    if(satellite == None): # If satellite ID wasn't supplied as an argument, use the satellite ID that is embedded in the image's metadata
        satellite = ds.GetMetadata().get('sat_id')
    dos_band_center = worldview_band_center(sat_id = satellite, band_num = dos_band) # Get the wavelength of the center of the DOS band from the band metadata
    scatteringFactor = dos_band_center**rayleighExp * dos_value # Compute scattering factor or DOS atmospheric correction
    
    dst_ds = gdal.GetDriverByName('GTiff').Create(output_fp, ds_cols, ds_rows, ds_nbands, gdal.GDT_Int16) # Create empty raster that will store atmospherically corrected pixel values
    dst_ds.SetGeoTransform(ds_geotransform) # Use the same geotranform as the input image
    dst_ds.SetProjection(ds_projection) # Use the same projection as the input image
    dst_ds.SetMetadata(ds_RPCs, 'RPC') # Use the same RPCs as the input image

    for j in range(0, ds_nbands): # Loop through each band of the input image
        band = ds.GetRasterBand(j+1) # Get the jth band
        col_bSize = band.GetBlockSize()[0] # Get the block size of the jth band in the col (x) dimension
        row_bSize = band.GetBlockSize()[1] # Get the block size of the jth band in the row (y) dimension
        band_center = worldview_band_center(sat_id = satellite, band_num = j) # Get the wavelength of center of the jth band
        atm_cor_val = scatteringFactor / band_center**rayleighExp # Compute atmospheric correction value to subtract from each pixel in the jth image image band
        for cpos in range(0, ds_cols, col_bSize): # Loop through blocks of the jth band
            if cpos + col_bSize < ds_cols:
                numCols = col_bSize
            else:
                numCols = ds_cols - cpos
            for rpos in range(0, ds_rows, row_bSize):
                if rpos + row_bSize < ds_rows:
                    numRows = row_bSize
                else:
                    numRows = ds_rows - rpos
                band_arr = band.ReadAsArray(cpos, rpos, numCols, numRows) # Read a block of the jth band
                mask = (band_arr != ds_nodata) # Find which pixels (i.e., elements in the block of the jth band) that contain no data
                atmCorr = np.full(np.shape(band_arr), ds_nodata) # Create an array containing the no data value with the same dimensions as the block of the jth band
                if mask.any(): # Proceed to atmospheric correction if there are valid pixels in the block of the jth band
                    atmCorr[mask] = band_arr[mask] - atm_cor_val # Sub
                    dst_ds.GetRasterBand(j+1).WriteArray(atmCorr, cpos, rpos)
                else: # If there are no valid pixels in the block of the jth band write no data
                    dst_ds.GetRasterBand(j+1).WriteArray(atmCorr, cpos, rpos)
        dst_ds.GetRasterBand(j+1).SetNoDataValue(ds_nodata) # Set the no data value for the jth band
    dst_ds.FlushCache() # Flush the output image cache to avoid read/write errors
    dst_ds = None
    ds.FlushCache() # Flush the input image cache to avoid read/write errors
    ds = None
    gc.collect()
##################### Mosaic a list of images
def mosaic(image_list, output_fp):
    create_directory(os.path.dirname(output_fp)) # Create output directory if it doesn't already exist
    input_list_txt = os.path.join(os.path.commonpath(image_list), 'tmp_tile_list.txt') # Create temporary txt file to store list of image files 
    with open(input_list_txt, 'w') as f: # Write list of image files to txt file
        for image in image_list:
            f.write(f"{image}\n")
    cmd = ["gdalbuildvrt", "-input_file_list", input_list_txt, output_fp] # Mosaic command
    subprocess.call(cmd) # Run mosaic command on the command line
    os.remove(input_list_txt) # Delete txt file
##################### Check if a polygonal shapefile is multipart of singlepart
def multipart_shp(shp_fp):
    ds = ogr.Open(shp_fp) # Open shapefile
    in_layer = ds.GetLayer() # Get layer
    feature = in_layer.GetNextFeature() # Get first feature
    while feature: # While feature doesn't equal None
        geometry = feature.GetGeometryRef() # Get geometry of feature
        geomType = geometry.GetGeometryName() # Get geometry type of feature
        feature = in_layer.GetNextFeature()  # Next feature
    ds = None # Close shapefile
    if geomType == 'MULTIPOLYGON': 
        return(True) # Return true if shapefile in multipart
    else:
        return(False) # Return false if shapefile is singlepart 
##################### Convert a multipart shapefile to a singlepart shapefile
def multipart_to_singlepart(shp_fp, out_fp, output_proj=None):
    ds = ogr.Open(shp_fp)  # Open input shapefile
    in_layer = ds.GetLayer()  # Get shapefile layer
    in_spatialref = in_layer.GetSpatialRef()  # Get the spatial reference of the input shapefile

    if os.path.exists(out_fp):  # Remove output file if it already exists
        os.remove(out_fp)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = driver.CreateDataSource(out_fp)  # Create output shapefile object

    if output_proj is None:  # If an output projection wasn't specified
        out_layer = dst_ds.CreateLayer('poly', in_spatialref, geom_type=ogr.wkbPolygon)  # Create polygon object
    else:  # If an output projection was specified
        out_layer = dst_ds.CreateLayer('poly', output_proj, geom_type=ogr.wkbPolygon)  # Create polygon object

    # Copy field definitions from input to output layer
    in_layer_defn = in_layer.GetLayerDefn()
    for i in range(in_layer_defn.GetFieldCount()):
        field_defn = in_layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)

    out_spatialref = out_layer.GetSpatialRef()  # Output spatial reference
    coordTrans = osr.CoordinateTransformation(in_spatialref, out_spatialref)  # Coordinate transformation between input shapefile and output shapefile

    for feature in in_layer:  # Loop through features
        geom = feature.GetGeometryRef()  # Get geometry of feature
        geom.Transform(coordTrans)  # Translate feature geometry to the spatial reference of the output shapefile

        if geom.GetGeometryName() == 'MULTIPOLYGON':  # If the input shapefile is a multipart polygon, split it into singlepart polygons
            for geom_part in geom:  # For each multipart polygon, create a singlepart polygon and copy it to the output shapefile
                out_feat = ogr.Feature(out_layer.GetLayerDefn())
                out_feat.SetGeometry(geom_part)
                for i in range(in_layer_defn.GetFieldCount()):
                    out_feat.SetField(i, feature.GetField(i))
                out_layer.CreateFeature(out_feat)
        else:  # Copy features to output shapefile
            out_feat = ogr.Feature(out_layer.GetLayerDefn())
            out_feat.SetGeometry(geom)
            for i in range(in_layer_defn.GetFieldCount()):
                out_feat.SetField(i, feature.GetField(i))
            out_layer.CreateFeature(out_feat)

    out_spatialref.MorphToESRI()  # Write the .prj file for the output shapefile
    prj_name = out_fp[:-4] + '.prj'
    prj = open(prj_name, 'w')
    prj.write(out_spatialref.ExportToWkt())
    prj.close()

    ds = None  # Close the input shapefile
    dst_ds = None  # Close the output shapefile


##################### Snap ROI boundary to raster grid and extract pixel values
def shp_to_roi(shp_fp, image_fp, output_dir, field_name = 'Classname'):
    raster_ds = gdal.Open(image_fp) # Open input image
    raster_nodata = raster_ds.GetRasterBand(1).GetNoDataValue()
    # raster_projection = raster_ds.GetProjection() # Get input image projection
    create_directory(output_dir) # Create output directory if it doesn't already exist
    vec_ds = ogr.Open(shp_fp) # Open input shapefile
    in_layer = vec_ds.GetLayer() # Get layer
    if field_name == None: # If the field name containing the class information wasn't supplied as an argument, prompt the user to specify it
        field_names = [field.name for field in in_layer.schema]
        print("Which of the following fields contains the class names or values of interest?")
        print(field_names)
        field_name = input("Field: ")
    field_vals = []
    feature = in_layer.GetNextFeature() # Get the first feature
    while feature: # Loop through features in shapefile
        field_vals.append(feature.GetFieldAsString(field_name)) # Get field values from each feature  
        feature = in_layer.GetNextFeature() 
    class_names = list(set(field_vals)) # List unique field values (class names)
    in_layer = None
    for unique_class in class_names: # Loop through each class
        in_layer = vec_ds.GetLayer() # Get shapefile layer
        vec_wkt = in_layer.GetSpatialRef().ExportToWkt() # Shapefile's spatial reference wkt string
        if unique_class is not None:
            in_layer.SetAttributeFilter(field_name + " = '" + str(unique_class) + "'") # Filter layer attributes to select features belonging to a specific class
        for i in range(0, in_layer.GetFeatureCount()): # Loop through filtered features
            feature = in_layer.GetNextFeature()
            vec_geom = feature.GetGeometryRef().GetEnvelope() # Create envelope geometry from feature geometry (this forces all ROIs to be rectangular)
            target = osr.SpatialReference() # Target spatial reference object
            target.ImportFromEPSG(4326) # Set the target spatial reference object to EPSG:4326 or lat,lon
            target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) # Force the target spatial reference object to except the traditional lon, lat ordering instead of lat, lon
            source = osr.SpatialReference(wkt = vec_wkt) # Shapefile spatial reference object
            transform = osr.CoordinateTransformation(source, target) # Create spatial reference transformation object
            line = ogr.Geometry(ogr.wkbLineString) # Create line geometry
            line.AddPoint(vec_geom[0], vec_geom[3]) # Add upper left coordinates as a point
            line.AddPoint(vec_geom[1], vec_geom[2]) # Add lower right coordinates as a point
            line.Transform(transform) # Transform the point object from the source to target spatial reference
            roi_extent = [] # empty list object to hold the roi extent coordinates 
            for pt in range(line.GetPointCount()): # Populate roi_extent list with ul_lon, ul_lat, lr_lon, lr_lat
                x,y = line.GetPoint(pt)[0:2]
                roi_extent.append(x)
                roi_extent.append(y)
            mask_pos = mask_extent(raster_ds, roi_extent)
            if(None not in mask_pos):
                ul_col, ul_row, lr_col, lr_row = mask_pos
                output_fp = os.path.join(output_dir, unique_class.replace(" ","_") + "_" + str(i) + ".TIF")
                cmd = ["gdal_translate", "-ot", "Int16", "-srcwin" , str(ul_col), str(ul_row), str(lr_col - ul_col), str(lr_row - ul_row), "-a_nodata", str(raster_nodata), "-eco", image_fp, output_fp] # Output image ROI using feature extent
                subprocess.call(cmd) # Run clip command on the command line
    in_layer = None # Remove layer
    raster_ds = None # Remove input image
    vec_ds = None # Remove input shapefile
##################### Get class names from ROI shapefile
def roi_classes(shp_fp, field_name = 'Classname'):
    vec_ds = ogr.Open(shp_fp) # Open input shapefile
    in_layer = vec_ds.GetLayer() # Get layer
    if field_name == None: # If the field name containing the class information wasn't supplied as an argument, prompt the user to specify it
        field_names = [field.name for field in in_layer.schema]
        print("Which of the following fields contains the class names or values of interest?")
        print(field_names)
        field_name = input("Field: ")
    field_vals = []
    feature = in_layer.GetNextFeature() # Get the first feature
    while feature: # Loop through features in shapefile
        field_vals.append(feature.GetFieldAsString(field_name)) # Get field values from each feature  
        feature = in_layer.GetNextFeature() 
    class_names = list(set(field_vals)) # List unique field values (class names)
    return(class_names) # Return list of unique class names
##################### Define DCNN model architecture
def dcnn_model(numChannels, dimension, numClasses):
    model = Sequential() # Define sequential model
    model.add(Convolution2D(32, (1, 1), activation='relu', input_shape=(dimension, dimension, numChannels))) # Add first layer
    model.add(Dropout(0.01))
    print(model.output)
    model.add(Convolution2D(16, (3, 3), activation='relu')) # Add second layer
    model.add(Dropout(0.01))
    print(model.output)
    model.add(Flatten()) # Flatten data volume into a 1D vector
    model.add(Dense(numClasses, activation='softmax')) # Classification operation
    print(model.output)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Learning parameters
    return model
##################### Train DCNN model
def train_dcnn(cnnFileName, training_data_directory, class_names, numChannels, dimension, selected_sample_per_class = 20000, balanced_option = 'balanced', epochs = 500, batchSize = 256, deleted_channels = []):
    save_directory = os.path.dirname(cnnFileName)
    create_directory(save_directory) # Create the specified output directory
    orig_stdout = sys.stdout # Create system output object 
    numClasses = len(class_names) # Number of classes
    multi_model = dcnn_model(numChannels, dimension, numClasses) # Create untrained model object
    patch_crop_point=int(np.floor(dimension/2)) # The start point to crop the ROIs
    labels = [] # Object to contain class labels 
    x_train = [] # Object to contain image data for training
    for class_numb in range(0, len(class_names)): # Loop through each class seperately 
        roi_list = glob.glob(os.path.join(training_data_directory, '*' + class_names[class_numb] + "*.TIF")) # Identify the image ROIs that exist per class 
        sample_data = [] # Object to contain sample data
        for patchnumb in range(0, len(roi_list)): # Loop through each ROI image
            ds = gdal.Open(roi_list[patchnumb]) # Read the ROI image
            ds_nodata = ds.GetRasterBand(1).GetNoDataValue()
            bands = [ds.GetRasterBand(i) for i in range(1, ds.RasterCount + 1)] # Initialize ROI image bands
            arr = np.array([band.ReadAsArray() for band in bands]) # Read ROI image as an array
            im = np.transpose(arr, [1, 2, 0]) # Transpose image array so that standard indexing can be used [cols, rows, bands]
            ds.FlushCache() # Flush ROI image cache to avoid read/write errors
            ds = None
            a_zeros = np.zeros([im.shape[0], im.shape[1]]) 
            if dimension > 1: # Discard the boundary pixels of the image
                a_zeros[0:, 0:patch_crop_point] = 1
                a_zeros[0:patch_crop_point, 0:] = 1
                a_zeros[-patch_crop_point:, 0:] = 1
                a_zeros[0:, -patch_crop_point:] = 1
            index_loc = np.where(a_zeros == 0)
            rows_loc = index_loc[0] # Number of rows
            cols_loc = index_loc[1] # Number of columns
            print('rows_loc cols_loc', im.shape[0], im.shape[1], cols_loc.shape[0], rows_loc.shape[0])
            data_divided_into = 1 # Training ROIs are divided into 1. This can be changed to increase the number of parts to improve memory constraint
            length_all_location = int(rows_loc.shape[0]) # Total number of samples within ROI
            division_len = int(np.ceil(length_all_location/data_divided_into))
            count_data_division = 0
            for iteration in range(0, length_all_location, division_len):
                print('code running')
                count_data_division = count_data_division + 1
                print('division number', count_data_division, '    :.....')
                print('division number', length_all_location, '    :.....', )
                if count_data_division == data_divided_into:
                    data_iter_end = length_all_location
                else:
                    data_iter_end = iteration + division_len
                data_length = (data_iter_end - iteration)
                f = np.zeros([data_length, dimension, dimension, im.shape[2]]) # Declare the testing variable with the number of samples, sample size, and number of bands
                image_index = np.zeros([data_length, 2])
                for data_iter in range(iteration, data_iter_end):
                    l = rows_loc[data_iter] # Sample row location
                    m = cols_loc[data_iter] # Sample col location
                    e = np.zeros([dimension, dimension,  im.shape[2]]) # Empty 3D array to store sample 
                    e[0:dimension, 0:dimension,0: im.shape[2]] = im[l -patch_crop_point:l+patch_crop_point+1, m -patch_crop_point:m+patch_crop_point+1, :] # Extract sample
                    image_index[(data_iter - iteration), :] = [l, m] # Save index information for sample location
                    f[(data_iter - iteration), :, :, :] = e # Add sample to list of samples
                null_pos = np.where(f == ds_nodata)
                if len(null_pos[0]) > 0:
                    f = f[~null_pos[0],:,:,:]
            sample_data.append(f) # Concatenate all samples from the same ROIs
        sample_data = np.concatenate(sample_data) # Concatenate all samples from all ROIs from the same class
        print('Sample data Shape', sample_data.shape)
        if balanced_option == 'balanced': # Randomly upsample or downsample to balance the data
            if np.shape(sample_data)[0] > selected_sample_per_class:
                index_downsample = np.random.choice(np.shape(sample_data)[0], selected_sample_per_class, replace=False)
            elif np.shape(sample_data)[0] < selected_sample_per_class:
                index_downsample = np.random.choice(np.shape(sample_data)[0], selected_sample_per_class)
            sample_data = sample_data[index_downsample, :, :, :]
            print('after blancing Sample data Shape', sample_data.shape)
            labels.append(class_numb * np.ones(selected_sample_per_class))
            x_train.append(sample_data)
            del sample_data
        elif balanced_option == 'unbalanced': # Unbalanced dataset
            labels.append(class_numb * np.ones(int(sample_data.shape[0])))
            x_train.append(sample_data)
            del sample_data
    x_train = np.concatenate(x_train) # Concatenate all the training samples
    labels = np.concatenate(labels) # Concatenate all of the labels 
    x_train = np.delete(x_train, deleted_channels, 3) # Delete bands from samples if the deleted_channels argument was specified
    print(x_train.shape, labels.shape)
    y_train = to_categorical(labels) # vector conversion
    f = open(os.path.join(save_directory, 'command_window'+'.txt'), 'w') # Write model training output to .txt file
    sys.stdout = f
    history = multi_model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize,validation_split=0.1, shuffle=True, verbose = 2) # Train model with ROI data. Split the labeled ROI data into train and test datasets, shuffle the training data before each epoch 
    print(history)
    sys.stdout = orig_stdout
    print(history)
    plt.clf() #  Save "accuracy" curve in png format for training and validation
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(save_directory,'accuracy.png'))
    plt.clf() # Save "loss" curve in png format for training and validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(save_directory,'loss.png'))
    plt.clf()
    with h5py.File(cnnFileName, mode = 'w') as f: # Save trained model
        multi_model.save(f)
        f.attrs['class_names'] = class_names # Save class names in the model's metadata
##################### Classify image with DCNN model
def dcnn_classification(image_fp, dcnn_fp, output_fp, bSize=256):
    create_directory(os.path.dirname(output_fp))  # Create output directory if it doesn't already exist
    
    # Read the specified dcnn model
    multi_model = tensorflow.keras.models.load_model(dcnn_fp)
    
    with h5py.File(dcnn_fp, mode='r') as f:
        class_names = f.attrs['class_names']  # Extract the class names from the model metadata

    dimension = list(multi_model.input_shape)[1]  # Extract the sample x, y dimensions
    patch_crop_point = int(np.floor(dimension / 2))  # Compute the crop point

   
    ds = gdal.Open(image_fp)  # Open input image
    ds_geotransform = ds.GetGeoTransform() # Get image geotransform
    ds_projection = ds.GetProjection() # Get image projection. If the image is using rational polynomial coefficients for spatial reference, this will be blank 
    ds_RPCs = ds.GetMetadata('RPC') # Get image rational polynomial coefficients. If the image is projected, this will be blank 
    ds_nbands = ds.RasterCount # Get the number of bands
    ds_cols = ds.RasterXSize # Get the number of columns (x)
    ds_rows = ds.RasterYSize # Get the number of rows (y)
    ds_nodata = ds.GetRasterBand(1).GetNoDataValue() # Get the no data value

    dst_ds = gdal.GetDriverByName('GTiff').Create(output_fp, ds_cols, ds_rows, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(ds_geotransform) # Use the same geotranform as the input image
    dst_ds.SetProjection(ds_projection) # Use the same projection as the input image
    dst_ds.SetMetadata(ds_RPCs, 'RPC') # Use the same RPCs as the input image
    dst_ds.GetRasterBand(1).SetNoDataValue(0) # Set no data value to zero
    label_list = ['']  # Create label list with empty first position to correspond to no data value of zero
    [label_list.append(i) for i in class_names] # Add class names to label list
    dst_ds.GetRasterBand(1).SetCategoryNames(label_list) # Add image classification labels to metadata

    bands = [ds.GetRasterBand(i) for i in range(1, ds_nbands + 1)] # Initialize input image band objects
    for cpos in range(patch_crop_point, ds_cols, bSize - (patch_crop_point*2)): # Loop through blocks of the input image
        if cpos + bSize < ds_cols:
            numCols = bSize
        else:
            numCols = ds_cols - cpos + patch_crop_point
        for rpos in range(patch_crop_point, ds_rows, bSize - (patch_crop_point*2)):
            if rpos + bSize < ds_rows:
                numRows = bSize
            else:
                numRows = ds_rows - rpos + patch_crop_point
            arr = np.array([band.ReadAsArray(cpos - patch_crop_point, rpos - patch_crop_point, numCols, numRows) for band in bands]) # Read block of input image
            mask = (arr != ds_nodata) # Determine where the no data values exist within the input image block
            if mask.any(): # Only proceed if the input image block contains valid pixels
                im = np.transpose(arr, [1, 2, 0]) # Transpose image array so that standard indexing can be used [cols, rows, bands]
                rows_loc = np.repeat(np.arange(patch_crop_point, (im.shape[0] - patch_crop_point)), im.shape[1] - (patch_crop_point*2)) # All row positions of samples within the input image block  
                cols_loc = np.concatenate([np.arange(patch_crop_point, (im.shape[1] - patch_crop_point))] * (im.shape[0] - (patch_crop_point*2))) # All column positions of samples within the input image block
                length_all_location = len(rows_loc) # Total number of samples within input image block
                f = np.zeros([length_all_location, dimension, dimension, ds_nbands]) # Declare the testing variable with the number of samples, sample size, and number of bands
                image_index = np.zeros([length_all_location, 2]) # Create object to store the image location index for each sample
                for data_iter in range (0, length_all_location): # Loop through input image block
                    l = rows_loc[data_iter] # Row position of sample
                    m = cols_loc[data_iter] # Column position of sample
                    e = np.zeros([dimension, dimension, ds_nbands])
                    e[0:dimension, 0:dimension, 0:ds_nbands] = im[(l-patch_crop_point):(l+patch_crop_point+1), (m-patch_crop_point):(m+patch_crop_point+1), 0:ds_nbands] # Extract sample from image block
                    image_index[(data_iter - 0), :] = [l, m] # Save the image row and column infomation for the sample
                    f[(data_iter - 0), :, :, :] = e # Add sample to list of samples
                predicted_label = multi_model.predict(f, batch_size=256) # Generate class probability estimates for each sample
                y_pred_arg = np.argmax(predicted_label, axis=1) # Select the class that has the highest probability estimate for each sample
                discrete_result = np.zeros([im.shape[0], im.shape[1]]) # Create array object with the same x,y dimensions as input image block 
                for index_predict in range(0, length_all_location): # Store classification values in their correct location
                    discrete_result[int(image_index[index_predict,0]), int(image_index[index_predict,1])] = y_pred_arg[index_predict] + 1 
                null_pos = np.where(f == ds_nodata) # Find samples that contain no data
                discrete_result[rows_loc[null_pos[0]], cols_loc[null_pos[0]]] = 0 # Change classification values to no data if they were computed from a sample that contained no data
                out = discrete_result[patch_crop_point:(im.shape[0]-patch_crop_point), patch_crop_point:(im.shape[1]-patch_crop_point)] # Crop the output classifiction array with patch crop point
                dst_ds.GetRasterBand(1).WriteArray(out, cpos, rpos) # Output discrete classification 
    ds.FlushCache() # Flush input image cache to avoid read/write errors
    ds = None 
    dst_ds.FlushCache() # Flush output image cache to avoid read/write errors                 
    dst_ds = None



