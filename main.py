from PIL import Image
import torch
import streamlit as st
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from MainModel import MainModel
from converter import lab_to_rgb
from torchvision import transforms

def gen_resunet(n_input=1, n_output=2, size=256):
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size))
    return net_G


st.title("Colour gans application")
st.write("")

# enable users to upload images for the model to make predictions
img_file = st.file_uploader("Upload an image", type="jpg")

gen = gen_resunet()
gen.load_state_dict(torch.load("colorcoco_final.pt",map_location=torch.device('cpu')))
model = MainModel(gen=gen)
model.load_state_dict(torch.load("model_gan.pt",map_location=torch.device('cpu')))

def predict(image):
    img = image.resize((256, 256))
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.gen(img.unsqueeze(0))
    pred_img = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    return pred_img


if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second ...")
    st.image(predict(image))

