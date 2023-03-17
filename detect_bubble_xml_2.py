import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, Model
import preproc_xml, glob
from PIL import Image
import numpy as np
# from tensorflow.keras import backend as bkd
# bkd.set_floatx('float32')
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
import random, timer, json, os, shutil

root_f = "."
ds_f = "./exds/psd_ds_2"
tr_f = "./exds/tr_2"
tt_f = "./exds/special_2"
model_f = "./exds/ver_2"

out_channels = 2
last_act = "softmax"
wh = (512, 768)
# https://www.tensorflow.org/tutorials/images/segmentation
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

def encode_bloc(X, nf, ks):
  X = layers.SeparableConv2D(nf, kernel_size = ks, padding="same", strides=(1, 1), activation = 'relu')(X)
  X = layers.SeparableConv2D(nf, kernel_size = ks, padding="same", strides=(1, 1), activation = 'relu')(X)
  # print(X.shape)
  return X

def max_p(X, sz):
  X = layers.MaxPooling2D(pool_size=sz, padding="same")(X)
  return X

def avg_p(X, sz):
  X = layers.AveragePooling2D(pool_size=sz, padding="same")(X)
  return X

def decode_bloc(X, Xn, nf, ks, std):
  X = layers.Conv2DTranspose(nf, kernel_size=ks, padding="same", strides=std, activation = 'relu')(X)
  X = layers.concatenate([Xn, X], axis=-1)
  X = encode_bloc(X, nf, ks)
  return X

def get_umodel(initial_bias): # 517,562
  input_shape = (wh[1], wh[0], 1)
  inputs = layers.Input(shape = input_shape,name='manga_input')
  
  X = encode_bloc(inputs,16, (6,4))
  X0 = X

  X = max_p(X,(2, 2))
  X = encode_bloc(X,32, (6,4))
  X1 = X

  X = max_p(X,(2, 2))
  X = encode_bloc(X,64, (3,2))
  X2 = X

  X = max_p(X,(2, 2))
  X = encode_bloc(X, 128, (3,2))
  X3 = X

  X = max_p(X,(2, 2))
  X = encode_bloc(X, 256, (3,2))

  X = decode_bloc(X, X3, 128, (3,2),(2, 2))
  X = decode_bloc(X, X2, 64, (3,2),(2, 2))
  X = decode_bloc(X, X1, 32, (6,4),(2, 2))
  X = decode_bloc(X, X0, 16, (6,4),(2, 2))

  predictions = layers.Conv2D(out_channels, kernel_size = (1,1), activation = last_act,bias_initializer=initial_bias)(X)
  model = models.Model(inputs=inputs,outputs=predictions)
  print(model.summary())
  return model

def im_to_a(p):
  with Image.open(p) as im:
    im = im.resize(wh).convert('L')
    a = np.expand_dims(np.array(im), axis=-1)
  return a

def mask_to_img(a, p):
  a = a *255
  # print(a)
  im = Image.fromarray(a.astype(np.uint8)).convert('L')
  im.save(p)
  return im

def get_n(b_y):
  r = 1
  for i in b_y.shape:
    r *= i
  return r

def get_avg_pos_neg():
  pms = glob.glob(f"{tr_f}\\*_m.jpg", recursive=False)
  b_y = np.zeros((wh[1], wh[0]))
  for i, pm in enumerate(pms):
    p = pm.replace("_m.jpg", ".jpg")

    am = im_to_a(pm)
    # print(am.shape)
    b_y += np.squeeze(am==255)
  
  pos = np.sum(b_y)/(i+1)
  tot = get_n(b_y)
  print(pos)
  print(tot)

  return pos, tot-pos

# 0 1 black, 1 0 white
def get_batch(bsz, diversed_color_ratio=0.4):
  pms = glob.glob(f"{tr_f}\\*_m.jpg", recursive=False)
  random.shuffle(pms)
  b_x = np.zeros((bsz, wh[1], wh[0], 1))
  b_y = np.zeros((bsz, wh[1], wh[0], out_channels))
  for i, pm in enumerate(pms[:bsz]):
    p = pm.replace("_m.jpg", ".jpg")
    diversed = random.random()<diversed_color_ratio
    a = im_to_a(p)
    if diversed:
      inversed_ratio = 0.6
      inversed = random.random() < inversed_ratio
      if inversed:
        a = (255 - a)
        print("color inversed")
      else:
        offset = int(random.random()*255)
        a = (offset - a)%255
        print(f"color diversed: {offset}")
      # print(a.shape)
    b_x[i] = a/255

    am = im_to_a(pm)
    # print(am.shape)
    # b_y[i] = am==255
    b_y[i]=np.concatenate((am==255, am==0), axis=-1)
    # print(np.sum(b_y[i]))
  # print(b_x)
  # print(b_y.shape)
  return b_x, b_y

def pred_tr(model):
  pms = glob.glob(f"{tr_f}\\*m.jpg", recursive=False)
  for i, pm in enumerate(pms):
    p = pm.replace("_m.jpg", ".jpg")
    ex_img = p
    ex_a = im_to_a(ex_img)/255
    ex_in=np.expand_dims(ex_a, axis=0)
    ex_mask = model.predict(ex_in)
    # print(type(ex_mask))
    ex_mask = np.squeeze(ex_mask)
    # print(ex_mask)
    # ex_mask = np.squeeze(ex_mask)
    # print(ex_mask.shape)
    ex_mask = np.argmin(ex_mask, axis=-1)
    # print(ex_mask.shape)
    # print(ex_mask)
    mask_p = pm.replace("_m.jpg", "_p.jpg")
    mask_to_img(ex_mask, mask_p)

def get_mnames():
  sp = f"{root_f}\\ds\\tt\\p_num_model_map.txt"
  if os.path.exists(sp):
    with open(sp, 'r') as f:
      names = f.readlines()
    return [n.strip() for n in names]
  else:
    return []

def pred_one_img(im, mask_p, model):
  ex_a = np.expand_dims(np.array(im), axis=-1)/255
  ex_in=np.expand_dims(ex_a, axis=0)
  ex_mask = model.predict(ex_in)
  # print(type(ex_mask))
  ex_mask = np.squeeze(ex_mask)
  # print(ex_mask)
  # ex_mask = np.squeeze(ex_mask)
  # print(ex_mask.shape)
  ex_mask = np.argmin(ex_mask, axis=-1)

  return mask_to_img(ex_mask, mask_p)

def pred_ex(model, mname):
  sp = f"{tt_f}\\p_num_model_map.txt"

  ps = glob.glob(f"{tt_f}\\*.jpg", recursive=False)
  ps += glob.glob(f"{tt_f}\\*.png", recursive=False)

  j = 1
  while True:
    p = f"{tt_f}\\021\\021.jpg"
    mask_p = p.replace(".jpg", f"_p{j}.jpg")
    if os.path.exists(mask_p):
      j += 1
    else:
      break

  for i, p in enumerate(ps):
    if "_p" in p:
      continue
    ex_img = p
    ex_a = im_to_a(ex_img)/255
    p = p[:-4] + ".jpg"
    ex_in=np.expand_dims(ex_a, axis=0)
    ex_mask = model.predict(ex_in)
    # print(type(ex_mask))
    ex_mask = np.squeeze(ex_mask)
    # print(ex_mask)
    # ex_mask = np.squeeze(ex_mask)
    # print(ex_mask.shape)
    ex_mask = np.argmin(ex_mask, axis=-1)
    # print(ex_mask.shape)
    # print(ex_mask)

    img_n = p.split('\\')[-1][:-4]
    if not os.path.exists(p[:-4]):
      os.mkdir(p[:-4])

    mask_p = p.replace(".jpg", f"\\{img_n}_p{j}.jpg")
    mask_to_img(ex_mask, mask_p)
  
  with open(sp, 'a') as f:
    f.write(mname+'\n')
  return True

def loss_mse_yp(y_true, y_pred):
  # tf.square
  diff = y_true - y_pred
  sqrd_dif = tf.square(diff) -tf.square(y_pred)
  ret = tf.reduce_mean(sqrd_dif, axis=-1)
  return ret

# https://stackoverflow.com/questions/51793737/custom-loss-function-for-u-net-in-keras-using-class-weights-class-weight-not
def weightedLoss(originalLossFunc, weightsList):
  pass

def merge_save_history(hs, h, model_path):
  l=len(h['loss'])
  for k,v in h.items():
    if len(v)!= l:
      hs[k]+=([0]*l)
    else:
      hs[k]+=v
  json.dump(hs, open(model_path+'.json', 'w'))
  return hs

def train(model=None):
  historys = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
  pos, neg = get_avg_pos_neg()
  initial_bias = tf.keras.initializers.HeNormal()# np.log(np.ones((750, 500))*(pos/neg))
  # print(f"initial_bias: {initial_bias}")
  total = pos + neg
  weight_for_0 = (1 / neg) * (total / 2.0)
  weight_for_1 = (1 / pos) * (total / 2.0)

  weight_list = [weight_for_0, weight_for_1]
  # print(weight_list)

  lr = 0.001 # 0.1 F, 0.01 2000epo10 only 200 300, 0.001 500 F, 
  opt = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-07,amsgrad=True)
  # lr = 0.01
  # opt = tf.keras.optimizers.SGD(
  #   learning_rate=lr, momentum=0.0, nesterov=False)
  # v3
  vname = "v4"
  #### IMPORTANT ###
  off_set = 12000 #
  epo = 5000
  epo_patience = 2

  if model == None:
    print("building models")
    model = get_umodel(initial_bias)
    loss = 'binary_crossentropy'#weightedLoss(tf.keras.losses.binary_crossentropy, weight_list)#'binary_crossentropy' 'mse' tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = ["accuracy"]# [tf.keras.metrics.MeanIoU(num_classes=2)] accuracy tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),"accuracy" 
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

  bsz = 16

  es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=epo_patience, verbose=2,restore_best_weights=True)
  model_path = model_f+f"\\{vname}_epo{off_set+1}to{off_set+epo}_bs{bsz}.h5"
  csvlg =tf.keras.callbacks.CSVLogger(model_path+'.csv', separator=",", append=True)
  cbks = [es, csvlg]
  
  for i in range(epo):
    print('#'*30,f"b{i+1}",'#'*30)
    b_X,b_y = get_batch(bsz)
    # if i == 0:
    #   mname = f"\\{vname}_{i+off_set+1}.h5"
    #   b_save_path = model_f+mname
    #   model.save(b_save_path)
    #   # historys = merge_save_history(historys, h.history, b_save_path[:-3])
    #   print("model saved")
    #   pred_ex(model, mname)

    h = model.fit(b_X, b_y, epochs=10, validation_split=0.5, verbose=1, batch_size=bsz, callbacks=cbks)
    if (i+1) % 100 == 0 or i == 0: #
      mname = f"\\{vname}_{i+off_set+1}.h5"
      b_save_path = model_f+mname
      assert not os.path.exists(b_save_path), "model path aready exists!!!!"
      # return
      model.save(b_save_path)
      historys = merge_save_history(historys, h.history, b_save_path[:-3])
      print("model saved")
      pred_ex(model, mname)
    t.lap()

  # pred_tr(model)


def pred_tt():
  ps = glob.glob(f"{model_f}\\*.h5", recursive=False)

  extrt_keys = lambda p: (int(p.split('_')[-2].split('ver\\v')[-1]), int(p.split('_')[-1].split('.')[0]))
  ps = sorted(ps, key=extrt_keys)

  mnames = get_mnames()
  for p in ps:
    mname = p.split('\\')[-1]
    if mname in mnames:
      # print("The model has already predicted!")
      pass
    else:
      re_model = tf.keras.models.load_model(p)
      pred_ex(re_model, mname)

def split_tr_tt():
  ps = glob.glob(f"{ds_f}\\*_m.jpg", recursive=False)
  random.seed(77)
  random.shuffle(ps)

  split_idx = int(0.8*len(ps))
  ps_tr = ps[:split_idx]
  ps_tt = ps[split_idx:]

  for p in ps_tr:
    shutil.copy(p, tr_f)
    shutil.copy(p[:-6]+".jpg", tr_f)
  for p in ps_tt:
    shutil.copy(p, tt_f)
    shutil.copy(p[:-6]+".jpg", tt_f)

if __name__ == "__main__":
 
  t = timer.Timer()
  t.start()

  # p = root_f+r"\ver\v2_400.h5"
  # re_model = tf.keras.models.load_model(p)
  # split_tr_tt()
  gpu_prex = '/job:localhost/replica:0/task:0/device:GPU:'
  with tf.device(gpu_prex+str(1)):
    # v4_200
    # v4_5400
    selected = "v4_6400.h5"
    model_name = r"E:\Research\manga\exds\ver_2\\"+ selected
    re_model = tf.keras.models.load_model(model_name)
    train(model=re_model)
  # pred_tt()
 
  
  t.stop()


