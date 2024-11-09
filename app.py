import torch
import cv2
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face):
    """
    Extract a 512-dimensional face embedding from an input face image.

    Args:
        face (numpy.ndarray): Cropped face image as a BGR numpy array.

    Returns:
        numpy.ndarray: 512-dimensional face embedding.
    """
    global face_model,device
    img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = face_model(img_tensor)  # Directly use the global face_model
    return embedding.squeeze().cpu().numpy()

# face_crop_image=cv2.imread("crop_face.jpg")
# embedding=get_embedding(face_crop_image)
# print(embedding)
