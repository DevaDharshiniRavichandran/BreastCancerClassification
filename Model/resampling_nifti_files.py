import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def sitk_resample(itk_image, out_spacing=[1.1,1.1,3.0], interpolation=None):
    # Getting the original attributes
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    resample = sitk.ResampleImageFilter()
    # Setting the output size
    out_size = [
                int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
                ]
    resample.SetSize(out_size)
    #  Setting the output spacing
    resample.SetOutputSpacing(out_spacing)
    # Setting the output direction
    resample.SetOutputDirection(itk_image.GetDirection())
    # Setting the output origin
    resample.SetOutputOrigin(itk_image.GetOrigin())
    # Setting the transform
    resample.SetTransform(sitk.Transform())
    # Setting the default pixel value
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    # Setting the interpolation
    if interpolation == None:
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolation == 'Linear':
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolation == 'NearestNeighbor':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation == 'BSpline':
        resample.SetInterpolator(sitk.sitkBSpline)
    else:
        raise Exception("The interpolator should be one of 'Linear', 'NearestNeighbor', 'BSpline'")
   
    return resample.Execute(itk_image)


def registration(df, output_folder, csv_path):
    for ind in tqdm(df.index.values.tolist()):
        path = df.loc[ind, 'NIFTI location path']
        # thickness = df.loc[ind, 'SliceThickness']
        # Reading the volume
        image = sitk.ReadImage(path)
        # Adding the pixel spacing as well
        pixel_spacing_x = image.GetSpacing()[0]
        pixel_spacing_y = image.GetSpacing()[1]
        pixel_spacing_z = image.GetSpacing()[2]
        print(pixel_spacing_x, pixel_spacing_y, pixel_spacing_z)
        df.loc[ind, 'PixelSpacing_X'] = pixel_spacing_x
        df.loc[ind, 'PixelSpacing_Y'] = pixel_spacing_y
        df.loc[ind, 'PixelSpacing_Z'] = pixel_spacing_z
        if pixel_spacing_y == pixel_spacing_z:
            out_spacing=[3.0,1.1,1.1]
        else:
            out_spacing=[1.1,1.1,3.0]

        # Resampling the volume
        resampled_image = sitk_resample(image, out_spacing, interpolation='BSpline')
        original_filename = os.path.basename(path)  # Extract filename from path
        resampled_filename = f"resampled_BSpline_{original_filename}"  # Prefix resampled_
        resampled_filepath = os.path.join(output_folder, resampled_filename)

        # Save the resampled image
        writer = sitk.ImageFileWriter()
        writer.SetFileName(resampled_filepath)
        writer.Execute(resampled_image)

        # Update the DataFrame with the new file path
        df.loc[ind, 'ResampledNIFTIPath'] = resampled_filepath


    df.to_csv(csv_path, index=False)

file_path = '/mnt/storage/deva/Data/ultimate.csv'
output_path = '/mnt/storage/deva/Data/Resampled_images_BSpline'
csv_output_path = '/mnt/storage/deva/Data/updated_ultimate_BSpline.csv'
df = pd.read_csv(file_path)
registration(df, output_path, csv_output_path)