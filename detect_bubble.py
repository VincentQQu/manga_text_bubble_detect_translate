import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, Model
import preproc, glob
from PIL import Image
import numpy as np
from tensorflow.keras import backend as bkd
bkd.set_floatx('float32')
import random, timer, json, os

root_f = "."
tr_f = "./ds/tr"
tt_f = "./ds/tt"
model_f = "./ver"

out_channels = 2
last_act = "softmax"
# https://www.tensorflow.org/tutorials/images/segmentation
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics



def get_smodel(initial_bias): #4,546
  input_shape = (750, 500, 1)
  inputs = layers.Input(shape = input_shape,name='manga_input')

  X = encode_bloc(inputs,16, (4,4))
  X = encode_bloc(X,32, (4,4))
  X = encode_bloc(X,16, (4,4))

  predictions = layers.Conv2D(out_channels, kernel_size = (1,1), activation = last_act,bias_initializer=initial_bias)(X)
  model = models.Model(inputs=inputs,outputs=predictions)
  print(model.summary())

  return model



def freeze(model):
  for layer in model.layers:
    layer.trainable = True




def encode_bloc(X, nf, ks):
  X = layers.SeparableConv2D(nf, kernel_size = ks, padding="same", strides=(1, 1), activation = 'relu')(X)
  X = layers.SeparableConv2D(nf, kernel_size = ks, padding="same", strides=(1, 1), activation = 'relu')(X)
  # print(X.shape)
  return X



def max_p(X, sz):
  X = layers.MaxPooling2D(pool_size=sz, padding="same")(X)
  return X



def decode_bloc(X, Xn, nf, ks, std):
  X = layers.Conv2DTranspose(nf, kernel_size=ks, padding="same", strides=std, activation = 'relu')(X)
  X = layers.concatenate([Xn, X], axis=-1)
  X = encode_bloc(X, nf, ks)
  return X



def get_umodel(initial_bias): # 167,994
  input_shape = (750, 500, 1)
  inputs = layers.Input(shape = input_shape,name='manga_input')
  
  X = encode_bloc(inputs,16, (6,4))
  X0 = X

  X = max_p(X,(2, 2))
  X = encode_bloc(X,32, (6,4))
  X1 = X

  X = max_p(X,(5, 5))
  X = encode_bloc(X,64, (3,2))
  X2 = X

  X = max_p(X,(5, 5))
  X = encode_bloc(X, 128, (3,2))

  X = decode_bloc(X, X2, 64, (3,2),(5, 5))
  X = decode_bloc(X, X1, 32, (6,4),(5, 5))
  X = decode_bloc(X, X0, 16, (6,4),(2, 2))

  predictions = layers.Conv2D(out_channels, kernel_size = (1,1), activation = last_act,bias_initializer=initial_bias)(X)
  model = models.Model(inputs=inputs,outputs=predictions)
  # print(model.summary())

  return model



# https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2
def get_mmodel(initial_bias): #1,494,031
  input_shape = (750, 500, 1)
  inputs = layers.Input(shape = input_shape,name='manga_input')
  X = layers.Conv2D(3, kernel_size = (6,4), padding="same", strides=(3, 2), activation = 'relu')(inputs)
  X0 = X
  X = layers.Conv2D(3, kernel_size = (27,27), padding="valid", strides=(1, 1), activation = 'relu')(X)

  base_model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3),include_top=False) #
  freeze(base_model)

  X = base_model(X)

  X = layers.Conv2DTranspose(128, kernel_size=(3,3), strides=(12, 7), output_padding=(0, 5), activation = 'relu')(X)
  X = layers.Conv2DTranspose(16, kernel_size=(3,2), strides=(5, 5), activation = 'relu')(X)
  X = layers.Conv2DTranspose(4, kernel_size=(3,2), padding="same", strides=(2, 2), activation = 'relu')(X)
  X = layers.concatenate([X, inputs], axis=-1)

  predictions = layers.Conv2D(out_channels, kernel_size = (1,1), activation = last_act,bias_initializer=initial_bias)(X) # sigmoid

  model = models.Model(inputs=inputs,outputs=predictions)
  # print(model.summary())
  # print(base_model.summary())
  return model



def im_to_a(p):
  with Image.open(p) as im:
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
  b_y = np.zeros((750, 500))
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
def get_batch(bsz):
  pms = glob.glob(f"{tr_f}\\*_m.jpg", recursive=False)
  random.shuffle(pms)
  b_x = np.zeros((bsz, 750, 500, 1))
  b_y = np.zeros((bsz, 750, 500, out_channels))
  for i, pm in enumerate(pms[:bsz]):
    p = pm.replace("_m.jpg", ".jpg")
    a = im_to_a(p)
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

  j = 0
  while True:
    p = f"{tt_f}\\0001\\0001.jpg"
    mask_p = p.replace(".jpg", f"_p{j}.jpg")
    if os.path.exists(mask_p):
      j += 1
      continue
    else:
      break

  for i, p in enumerate(ps):
    if "_p" in p:
      continue
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

    idx = p.find(".jpg")
    img_n = p[idx-4:idx]
    if not os.path.exists(p[:idx]):
      os.mkdir(p[:idx])

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
  print(weight_list)

  lr = 0.01 # v1: 0.001, v2: 0.01
  opt = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-07,amsgrad=True)
  # v1: um
  # v2: mm
  vname = "v2"
  off_set = 351
  epo = 1000
  if model == None:
    model = get_mmodel(initial_bias)

  loss = 'binary_crossentropy'#weightedLoss(tf.keras.losses.binary_crossentropy, weight_list)#'binary_crossentropy' 'mse' tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  metrics = ["accuracy"]# [tf.keras.metrics.MeanIoU(num_classes=2)] accuracy tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),"accuracy" 
  model.compile(optimizer=opt, loss=loss, metrics=metrics)

  bsz = 8
  
  for i in range(epo):
    print('#'*30,f"b{i}",'#'*30)
    b_X,b_y = get_batch(bsz)

    h = model.fit(b_X, b_y, epochs=5, validation_split=0.2, verbose=1, batch_size=bsz)
    if (i+1) % 50 == 0:
      b_save_path = f"ver\\{vname}_{i+off_set}.h5"
      model.save(b_save_path)
      historys = merge_save_history(historys, h.history, b_save_path[:-3])
    t.lap()

  pred_tr(model)




def pred_tt():
  ps = glob.glob(f"{model_f}\\*.h5", recursive=False)

  extrt_keys = lambda p: (int(p.split('_')[-2].split('ver\\v')[-1]), int(p.split('_')[-1].split('.')[0]))
  ps = sorted(ps, key=extrt_keys)
  # print(ps)
  # return

  mnames = get_mnames()
  for p in ps:
    mname = p.split('\\')[-1]
    if mname in mnames:
      # print("The model has already predicted!")
      pass
    else:
      re_model = tf.keras.models.load_model(p)
      pred_ex(re_model, mname)




if __name__ == "__main__":
 
  t = timer.Timer()
  t.start()

  # p = root_f+r"\ver\v2_400.h5"
  # re_model = tf.keras.models.load_model(p)
  train(model=None)
  pred_tt()
 
  
  t.stop()


