from re import X
from PIL import Image, ImageDraw
import glob, timer, json, os
import numpy as np
import xml.etree.ElementTree as ET
# https://docs.python.org/3/library/xml.etree.elementtree.html

ds_f = r"E:\Research\manga\exds\Manga109\images"
label_f = r"E:\Research\manga\exds\Manga109\annotations"
names_p = r"E:\Research\manga\exds\Manga109\books.txt"
folder = ""

wh = (512, 768)
manga_sz = (1658, 1170)
# width=750 
# height=500

export_f = r"E:\Research\manga\exds\psd_ds_2"


def png_to_jpg(ps):
  for p in ps:
    im = Image.open(p)
    rgb_im = im.convert('RGB')
    rgb_im.save(f'{p[:-4]}.jpg')

def get_img_ps(folder, ext="jpg"):
  ps = glob.glob(f"{folder}**\\*.{ext}", recursive=True)
  # print(ps)
  return ps

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

def get_title_name(p):
  title = p.split('\\')[-2]
  name = p.split('\\')[-1][:-4]
  return title, name

def get_boxes_from_xml(p_ann):
  tree = ET.parse(p_ann)
  root = tree.getroot()

  rects = []
  pages = root.find('pages')
  for i, page in enumerate(pages):
    rects.append([])
    bs = page.findall("text")
    for b in bs:
      rect = (int(b.get("xmin")), int(b.get("ymin")), int(b.get("xmax")), int(b.get("ymax")))
      rects[i].append(rect)
  return rects

def cut_resize(img):
  # (left, top, right, bottom)
  img_l = img.crop((0, 0, manga_sz[0]//2, manga_sz[1]))
  img_r = img.crop((manga_sz[0]//2+1, 0, manga_sz[0], manga_sz[1]))
  rsz_img_l = img_l.resize(wh)
  rsz_img_r = img_r.resize(wh)
  return rsz_img_l, rsz_img_r

# https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
def save_masks(bboxes, prefix):
  

  for i, page_b in enumerate(bboxes):
    print(f"p{i}")
    img = Image.new('L', manga_sz) #, (0)
    draw = ImageDraw.Draw(img)
    for b in page_b:
      # print(b)
      draw.rectangle(b, fill="white")

    rsz_img_l, rsz_img_r = cut_resize(img)
    rsz_img_l.save(prefix+"{:03d}_l_m.jpg".format(i))
    rsz_img_r.save(prefix+"{:03d}_r_m.jpg".format(i))

  
def resize_save(ps, w=wh[0], h=wh[1]):
  for p in ps:
    with Image.open(p) as im:
      img = im.convert('L')
    rsz_img_l, rsz_img_r = cut_resize(img)
    title, name = get_title_name(p)
    sp = export_f+f"\\{title}_{name}"
    rsz_img_l.save(sp+"_l.jpg")
    rsz_img_r.save(sp+"_r.jpg")

def export_rsz_img_mask():
  if not os.path.exists(export_f):
    os.mkdir(export_f)

  with open(names_p) as f:
    book_names = f.readlines()
  for bn in book_names:
    bn = bn.strip()
    print(bn)
    print("saving imgs...")
    bf = ds_f+f"\\{bn}"
    ps = get_img_ps(bf, ext="jpg")
    resize_save(ps)
    print("saving masks...")
    p_ann = label_f+f"\\{bn}.xml"
    bboxes = get_boxes_from_xml(p_ann)
    prefix = export_f+f"\\{bn}_"
    save_masks(bboxes, prefix)
  

if __name__ == "__main__":
  t = timer.Timer()
  t.start()

  
  # ps_png = get_img_ps(folder, ext="png")
  # png_to_jpg(ps_png)
  
  export_rsz_img_mask()
  # resize(ps)

  # jfile = folder+r"\boxs.json"
  # boxes_to_mask(jfile)
  # rotate_flip()

  # for i in range(2):
  #   pass
  # print(i)
  

  t.stop()