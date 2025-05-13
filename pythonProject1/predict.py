from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/Users/mac/data/workspace/ultralytics-Satellite/runs/segment/train14/weights/best.pt')
    model.predict(source='/Volumes/My Passport/dataset/UESD/test/images',imgsz=1280,project='runs/segment/feature',
                  name='yolov11-seg-improved', save= True,visualize = True)
