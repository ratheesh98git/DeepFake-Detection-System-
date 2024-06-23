import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms

class LSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_backward_hook(hook_function)
        self.target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        class_loss = model_output[0, target_class]
        class_loss.backward()
        guided_gradients = self.gradients.cpu().data.numpy()[0]
        target = self.activations.cpu().data.numpy()[0]
        weights = np.mean(guided_gradients, axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_image.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def extract_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def extract_audio_features(video_path):
    y, sr = librosa.load(video_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs.T

def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(frame).unsqueeze(0)

video_path = 'c:/ratheesh/download.mp4'
frames = extract_video_frames(video_path)
audio_features = extract_audio_features(video_path)
cnn_model = resnet50(pretrained=True)
lstm_model = LSTMDetector(input_size=512, hidden_size=128, num_layers=2, num_classes=2)
frame_embeddings = []
for frame in frames:
    frame_tensor = preprocess_frame(frame)
    embedding = cnn_model(frame_tensor)
    frame_embeddings.append(embedding)
frame_embeddings = torch.stack(frame_embeddings).squeeze(1).unsqueeze(0)
output = lstm_model(frame_embeddings)
predicted_class = torch.argmax(output, dim=1).item()
grad_cam = GradCAM(cnn_model, cnn_model.layer4[2])
explanation = grad_cam.generate_cam(preprocess_frame(frames[0]), predicted_class)
print("Predicted Class:", "Real" if predicted_class == 0 else "Fake")
explanation = cv2.applyColorMap(np.uint8(255 * explanation), cv2.COLORMAP_JET)
frame_with_cam = np.float32(explanation) + np.float32(frames[0])
frame_with_cam = 255 * frame_with_cam / np.max(frame_with_cam)
cv2.imshow("Explanation", np.uint8(frame_with_cam))
cv2.waitKey(0)
