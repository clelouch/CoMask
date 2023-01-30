# this script is created to generate and save the prediction result using existing pretrained models

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
import cv2

# Specify the path to model config and checkpoint file
config_file = 'configs/CoMask/CoMask_r50_mfpn_2x.py'
checkpoint_file = './checkpoints/epoch_30.pth'


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
root_path = './Images'
output_path = './output/CoMask'

images = os.listdir(root_path)
for name in images:
    img = os.path.join(root_path, name)
    result = inference_detector(model, img)
    # # visualize the results in a new window
    # model.show_result(img, result)
    # or save the visualization results to image files

    bbox_result, segm_result = result
    print(segm_result[1])
    show_result_pyplot(img, result, model.CLASSES, out_file=os.path.join(output_path, name[:-4] + '.png'))
