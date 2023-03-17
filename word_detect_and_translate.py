
from PIL import Image
import pytesseract
from googletrans import Translator
import detect_bubble
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import timer, re


# modify here to youir tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

width=512
height=768
translator = Translator()
t = timer.Timer()
# p9: v1_450.h5 best
# 87: v1_4350.h5 third
# 88: v1_4400.h5 second
selected = "v4_13600.h5"

model_fdr = "./exds/ver_2/"
mname = model_fdr + selected
img_f = f"./wd_lab/"+selected[:-3]+'/'
print(img_f)
# print(img_f)
# with detect_bubble.tf.device('/device:GPU:1'):
re_model = detect_bubble.tf.keras.models.load_model(mname)
# img -> resized -> mask -> upsmp_mask
# get contours from upsmp_mask
# composite img and upsmp_mask as combo
# iterrate contours in combo and translate



def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key, reverse=False)



def imgs_to_txt(folder, ext="jpg", one_book=True):
  ps = glob.glob(f"{folder}**/*.{ext}", recursive=True)
  ps_png = glob.glob(f"{folder}**/*.png", recursive=True)

  ps = [ppg.replace("\\", "/") for ppg in ps]
  ps_png = [ppg.replace("\\", "/") for ppg in ps_png]

  
  ps += ps_png
  ps = sorted_alphanumeric(ps)
  print(ps)
  for p in ps:
    print(p)
    tmp_f = p[:-4]
    if not os.path.exists(tmp_f):
      os.mkdir(tmp_f)
    img_name = p.split('/')[-1][:-4]
    tmp_n = tmp_f +'/'+ img_name
    img = Image.open(p)
    # if p[-4:] == ".png":

    p_rsz = tmp_n+"_rsz.jpg"
    rsz_img = img.resize((width, height)).convert('L')
    rsz_img.save(p_rsz)
    p_m = tmp_n+"_m.jpg"

    
    mask = detect_bubble.pred_one_img(rsz_img, p_m, re_model)
    upsmp_mask = mask.resize(img.size, resample=Image.NEAREST)
    p_um = tmp_n+"_um.jpg"
    upsmp_mask.save(p_um)

    upsmp_mask_cv = cv2.imread(p_um)
    gray = cv2.cvtColor(upsmp_mask_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)
    img_cv = cv2.imread(p)

    # image_cv = cv2.drawContours(img_cv, contours, -1, (0, 255, 0), 2)
    # p_cont = tmp_n+"_cont1.jpg"
    # cv2.imwrite(p_cont, image_cv)

    # https://stackoverflow.com/questions/55362501/drawing-contours-with-bounded-rectangles-on-existing-image
    rects = []
    frist_rec = (0, 0)
    for contour in contours:
      # Find bounding rectangles
      x,y,w,h = cv2.boundingRect(contour)
      if frist_rec == (0, 0):
        frist_rec = (w,h)
        continue
      if w*h < 0.001*frist_rec[0]*frist_rec[1]: # w < 30 or h < 50 or 
        continue
      rects.append([(x,y),(x+w,y+h)])
      # Draw the rectangle
      
    
    area_order = lambda e: -(e[1][0]-e[0][0]) * (e[1][1]-e[0][1])
    rects = sorted(rects, key=area_order)
    filled_area = np.zeros(frist_rec)
    i = 0
    while i < len(rects):
      xy0, xy1 = rects[i]
      contain_zero = filled_area[xy0[0]:xy1[0], xy0[1]:xy1[1]] == 0
      if np.sum(contain_zero) > 10:
        filled_area[xy0[0]:xy1[0], xy0[1]:xy1[1]] = 1
        i += 1
        cv2.rectangle(img_cv,(xy0[0],xy0[1]),(xy1[0],xy1[1]),(255,255,0),2)
      else:
        rects.pop(i)
      

    jp_order = lambda e: e[0][1]-0.2*e[0][0]
    rects = sorted(rects, key=jp_order)

    # plt.imshow(image)
    # plt.show()

    # print(rects)
    content = p+"\n"
    
    # print(rects)
    for i, r in enumerate(rects, 1):
      xy0, xy1 = r
      box = (*xy0, *xy1)
      cv2.putText(img_cv, str(i), (xy0[0], xy0[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), thickness=6)
      content += (f"[b{i}]\n")
      bubble = img.crop(box=box)
      # print(bubble.size*2)
      factor = 2
      sz = (int(bubble.width*factor), int(bubble.height*factor))
      print(sz)
      bubble = bubble.resize(sz, Image.LANCZOS)
      bubble.save(tmp_n+f"_b{i}.jpg")
      content += trans(bubble)
    

    if one_book:
      book_fdr = '/'.join(p.split('/')[:-1])
      book_fdr += "/#cont/"
      if not os.path.exists(book_fdr):
        os.mkdir(book_fdr)
      p_cont = book_fdr+img_name+"_cont.jpg"
      cv2.imwrite(p_cont, img_cv)
      with open(book_fdr+"trans.txt", 'a') as f:
        f.write(content)
    else:
      p_cont = tmp_n+"_cont.jpg"
      cv2.imwrite(p_cont, img_cv)
      
      with open(p[:-4]+".txt", 'w') as f:
        f.write(content)
    
    t.lap()



def rm_sp_chr(string, chrs=None):
  if chrs == None:
    chrs = [' ', '\\', '/','・', '：' ]
    # unics = []
  for c in chrs:
    string = string.replace(c, '')
  return string



def tran_two_way(string, string_1, result, lang):
  translated = translator.translate(string, src='ja', dest=lang) # dest=zh-cn zh-tw
  result += ('+|'+translated.text.strip()+'|+\n')
  


  translated_1 = translator.translate(string_1, src='ja', dest=lang) # dest=zh-cn zh-tw
  if translated.text.strip() != translated_1.text.strip():
    result += ('-'*20+'\n')
    result += ('+|'+translated_1.text.strip()+'|+\n')
  
  return result


# https://www.thepythoncode.com/article/contour-detection-opencv-python
# https://note.nkmk.me/en/python-pillow-composite/
def trans(im):
  # print(pytesseract.get_languages())

  result = ""
  string = pytesseract.image_to_string(im, lang="jpn_vert")
  string = string.strip()
  string = rm_sp_chr(string)
  string = re.sub(r'\n+', '\n', string)
  string_1 = string.replace('\n', '')
  
  if string != None and string != '':
    result += ('-|'+string+'|-\n')
    result += ('-'*20+'\n')
    
    try:
      result = tran_two_way(string, string_1, result, "zh-cn")
    except Exception:
      result += "Translater Error: zh-cn\n"
    result += ('+'*20+'\n')
    try:
      result = tran_two_way(string, string_1, result, "en")
    except Exception:
      result += "Translater Error: en\n"
    
    
    # result += ('-'*20+'\n')

    # translated = translator.translate(string, src='ja', dest="zh-tw") # dest=zh-cn zh-tw
    # result += ('+|'+translated.text.strip()+'|+\n')
    # result += ('-'*20+'\n')

    # translated = translator.translate(string, src='ja', dest="en")
    # result += ('+|'+translated.text.strip()+'|+\n')
    # print('='*20)
    # boxes = pytesseract.image_to_boxes(img, lang="jpn")
    # print(boxes)
  else:
    print("unrecognisable")

  result += ('='*20+'\n')
  print(result)
  return result



def tran_sent():
  s = "なんだぁ?"
 
  translated = translator.translate(s, src='ja', dest="zh-cn")
  print(s)
  print(translated.text)



if __name__ == "__main__":
  
  t.start()
  # img_f = r"E:\Research\manga\original\\"
  
  imgs_to_txt(img_f, ext="jpg")
  # mname = r"E:\Research\manga\ver\v1_4350.h5"
  # re_model = detect_bubble.tf.keras.models.load_model(mname)
  # img_f = r"E:\Research\manga\wd_lab\\v1_4350\\"
  # imgs_to_txt(img_f, ext="jpg")
  # mname = r"E:\Research\manga\ver\v1_4400.h5"
  # re_model = detect_bubble.tf.keras.models.load_model(mname)
  # img_f = r"E:\Research\manga\wd_lab\\v1_4400\\"
  # imgs_to_txt(img_f, ext="jpg")
  # tran_sent()
  t.stop()
