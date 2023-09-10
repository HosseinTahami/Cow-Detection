import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from google.colab.patches import cv2_imshow

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the video
video_path = "/content/video6001248409883774665.mp4"
cap = cv2.VideoCapture(video_path)

# Define the classes for object detection
class_names = ["cow"]

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img_tensor = F.to_tensor(frame)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(img_tensor)

    # Check if any predictions were made
    if len(predictions) > 0:
        # Process the predictions
        for prediction in predictions[0]["boxes"]:
            class_index = int(predictions[0]["labels"][0])
            if class_index < len(class_names):
                class_name = class_names[class_index]
                if class_name == "cow":
                    x_min, y_min, x_max, y_max = prediction.tolist()
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        class_name,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

    # Write the modified frame to the output video
    out.write(frame)

    # Display the frame with bounding boxes
    cv2_imshow(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture, video writer, and close the windows
cap.release()
out.release()
cv2.destroyAllWindows()
