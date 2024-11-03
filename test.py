from model import *
import torch
from PIL import Image
from torchvision import transforms

def predict(image_path, is_gpu=False):
    if is_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print('Loadinging model...\n')
    model = simpleUNet(n_classes=2).to(device)  # 调包用的功能

    model.load_state_dict(  # 载入此前已经训练好的模型
        torch.load('saved/cont/best_checkpoint.pth', map_location=str(device))['state_dict'])
    model.eval()

    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose(
        [transforms.Resize([100, 100]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485], std=[0.229])
         ])
    image = transform(image)
    input = image.to(torch.float32)
    input = input.unsqueeze(0)
    with torch.no_grad():
        if str(device) != 'cpu':
            input = input.cuda()
        output = model(input)
        pred = torch.argmax(output, dim=1)
        prediction = pred.item()
    if prediction == 1:  # 说明患病了
        return True
    else:  # 说明没有患病
        return False

if __name__ == "__main__":
    out = predict("..\\data\\test\\COVID\\Covid (503).png", is_gpu=False)
    print(out)
