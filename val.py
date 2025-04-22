from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/segment/train9/weights/best.pt')
    metrics = model.val()



    print(metrics)