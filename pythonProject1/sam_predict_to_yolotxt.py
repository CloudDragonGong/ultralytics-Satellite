from ultralytics.data.annotator import auto_annotate


if __name__ == '__main__':
    auto_annotate(data='/Volumes/My Passport/dataset/UESD/yolov11/val/images',
                  det_model='/Users/mac/data/workspace/ultralytics-Satellite/runs/segment/train14/weights/best.pt',
                  sam_model='sam_b.pt',
                  imgsz=1280,
                  device='mps',
                  output_dir='/Volumes/My Passport/dataset/UESD/yolov11/val/sam_labels_yolo')