from model import Vixen
from PIL import Image


model = Vixen(ckpt='weights/vixen-c.ckpt').to(0).eval()

im1 = Image.open('0000014/10483310_0.jpg')
im2 = Image.open('0000014/10483310_1.jpg')

model.caption(im1, im2)
