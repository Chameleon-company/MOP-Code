from utils import draw_BOX,get_BOXes_from_sahi_result,get_BOXes_from_yolo_result,get_sliced_prediction,merge_overlapping_boxes,merge_img_horizen
import cv2
from ultralytics import YOLO

## get yolo with sahi prediction
def get_sahi_prediction(image_path, detection_model, slice_height=256, slice_width=256, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
    """
    Generates predictions for an image using the SAHI (Slicing Aided Hyper Inference) method and draws bounding boxes on the image.

    Args:
        image_path (str): Path to the input image.
        detection_model: The detection model to be used for predictions.
        slice_height (int, optional): Height of the slices. Defaults to 256.
        slice_width (int, optional): Width of the slices. Defaults to 256.
        overlap_height_ratio (float, optional): Overlap ratio for height between slices. Defaults to 0.2.
        overlap_width_ratio (float, optional): Overlap ratio for width between slices. Defaults to 0.2.

    Returns:
        image: The image with bounding boxes drawn
    """
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    res = cv2.imread(image_path)
    Box_list = list(get_BOXes_from_sahi_result(result))
    Box_list = merge_overlapping_boxes(Box_list)
    for Box in Box_list:
        res = draw_BOX(res,Box)

    cv2.putText(res, f'sahi: detected {len(Box_list)} items.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    return res
    
def multi_scale_sahi_prediction(image_path, detection_model,start_size=128,end_size=512,step=64):
    """
        Perform multi-scale prediction on an image using the SAHI (Slicing Aided Hyper Inference) method.
        Args:
            image_path (str): Path to the input image.
            detection_model: The detection model to be used for predictions.
            start_size (int, optional): The starting size of the slices. Default is 128.
            end_size (int, optional): The ending size of the slices. Default is 512.
            step (int, optional): The step size for increasing the slice size. Default is 64.
        Returns:
            image: The image with bounding boxes drawn
    """
    res = cv2.imread(image_path)
    Box_list = []
    for size in range(start_size,end_size,step):
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=size,
            slice_width=size,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,

        )
        tmp_box_list = list(get_BOXes_from_sahi_result(result))
        
        Box_list.extend(list(get_BOXes_from_sahi_result(result)))
    Box_list = merge_overlapping_boxes(Box_list)
    for Box in Box_list:
        Box.merge_label()
        res = draw_BOX(res,Box)

    cv2.putText(res, f'sahi_multi_scale: detected {len(Box_list)} items.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    return res




def get_yolo_prediction(image_path, yolo_model:YOLO):
    """
        Generates YOLO predictions for a given image and returns the image with bounding boxes drawn.

        Args:
            image_path (str): The file path to the input image.
            yolo_model (YOLO): An instance of the YOLO model used for making predictions.

        Returns:
            numpy.ndarray: The image with bounding boxes drawn around detected objects and a text label indicating the number of detected items.
    """
    res = cv2.imread(image_path)
    result = yolo_model.predict(res)
    Box_list = list(get_BOXes_from_yolo_result(result[0]))
    Box_list = merge_overlapping_boxes(Box_list)
    for Box in Box_list:
        res = draw_BOX(res,Box)
    cv2.putText(res, f'yolo: detected {len(Box_list)} items.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return res

def test_performance(input_path,output_path,detection_model,yolo_model,slice_height=256, slice_width=256, overlap_height_ratio=0.2, overlap_width_ratio=0.2):
    res_sahi_multi = multi_scale_sahi_prediction(
        input_path,detection_model,start_size=128,end_size=512
    )
    res_sahi = get_sahi_prediction(
        input_path,detection_model,slice_height=slice_height, slice_width=slice_width, overlap_height_ratio=overlap_height_ratio, overlap_width_ratio=overlap_width_ratio
    )
    res_yolo = get_yolo_prediction(
        input_path,yolo_model
    )
    res = merge_img_horizen(res_sahi,res_yolo)
    res = merge_img_horizen(res_sahi_multi,res)
    cv2.imwrite(output_path,res)