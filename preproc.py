from re import X
from PIL import Image
import glob, timer, json
import numpy as np

folder = "./ds/tt"
width=500
height=750


def png_to_jpg(ps):
  for p in ps:
    im = Image.open(p)
    rgb_im = im.convert('RGB')
    rgb_im.save(f'{p[:-4]}.jpg')

def get_img_ps(folder, ext="jpg"):
  ps = glob.glob(f"{folder}**/*.{ext}", recursive=True)
  # print(ps)
  return ps

def resize(ps, w=width, h=height):
  for p in ps:
    with Image.open(p) as im:
      im_resized = im.resize((w, h)).convert('L')
    im_resized.save(p)

def boxes_to_mask(jfile):
  with open(jfile) as json_file:
    data = json.load(json_file)
    # print(data)
  
  for i, boxes in enumerate(data, 1):
    mask_name = folder+"\{:04d}_m.jpg".format(i)
    ni = np.zeros((height, width), dtype=np.uint8)
    # np.uint8
    for b in boxes:
      x_s, y_s, w, h = b
      x_e = x_s + w
      y_e = y_s + h
      # print(x_s, x_e, y_s, y_e)
      x = x_s
      while x < x_e:
        y = y_s
        while y < y_e:
          # print(x_s, y_s)
          ni[y, x] = 255
          y += 1
        x += 1
    Image.fromarray(ni).convert('L').save(mask_name)
    # with Image.open(mask_name) as im:
    #   a = np.array(im)
    #   print(a.shape)
    # Image.fromarray(ni).convert('RGB').save(mask_name)

def rotate_flip():
  # read_dir = dataset_root+ dn+"-reduce-angular" + s
  # read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  rotate_angles = [0,90,180,270]
  for i in range(1, 11):
    pm = folder+"\{:04d}_m.jpg".format(i)
    p = pm.replace("_m.jpg", ".jpg")
    print('processing',p,'...')
    im = Image.open(p)
    imm = Image.open(pm)
    for r in rotate_angles:
      img = im.rotate(r)
      new_name = pm.replace("_m.jpg", f"r{r}.jpg")
      img.save(new_name)
      img_f = img.transpose(Image.FLIP_LEFT_RIGHT)
      new_name_f = pm.replace("_m.jpg", f"r{r}f.jpg")
      img_f.save(new_name_f)

      imgm = imm.rotate(r)
      new_name = pm.replace("_m.jpg", f"r{r}_m.jpg")
      imgm.save(new_name)
      img_f = imgm.transpose(Image.FLIP_LEFT_RIGHT)
      new_name_f = pm.replace("_m.jpg", f"r{r}f_m.jpg")
      img_f.save(new_name_f)
    im.close()
    imm.close()
  print('success! - rotate_flip')


if __name__ == "__main__":
  t = timer.Timer()
  t.start()

  
  # ps_png = get_img_ps(folder, ext="png")
  # png_to_jpg(ps_png)
  ps = get_img_ps(folder, ext="jpg")
  resize(ps)

  # jfile = folder+r"\boxs.json"
  # boxes_to_mask(jfile)
  # rotate_flip()

  # for i in range(2):
  #   pass
  # print(i)
  

  t.stop()


  

