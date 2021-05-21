# -*- coding: utf-8 -*-
"""
@author: 김도윤
"""
# -*- coding: utf-8 -*-

################################
first_file_no = 50                                                            
last_file_no = 85
test_image_no = 74                                                  
size_hf = 14                                                                                                                                 
seed = 0                                                                      
###############################

"""train X, Y 만들기"""

import numpy as np
import pickle


with open('data.pickle', 'rb') as file:
    dictionary = pickle.load(file)
    
prev_len=0
total_len = 0
for n in range (first_file_no, last_file_no+1):
  
  if n == test_image_no:
    continue
  
  elif n in dictionary:
    marker_x = dictionary[n]['marker_x']
    marker_y = dictionary[n]['marker_y']

#################################################333    
#    total_len += (len(marker_x))*2

    prev_len += (len(marker_x))*2
    total_len += (len(marker_x)*25 )* 2 #24확장
###############################################

  else:
      continue

X = np.zeros((total_len, size_hf*2, size_hf*2, 3)) * np.nan
Y = np.zeros((total_len, 2)) * np.nan
Z = np.zeros(total_len) * np.nan

last_num = 0

##%

for n in range (first_file_no, last_file_no+1):
#마커, 이미지 정보 불러오기
    if n == test_image_no:
      continue
    
    elif n in dictionary:
        marker_x = dictionary[n]['marker_x']
        marker_y = dictionary[n]['marker_y']
        width = dictionary[n]['width']
        length = dictionary[n]['length']
        im = dictionary[n]['imread']


###############
       #  marker 방향 확장
        for i in range(0,len(marker_x)):
          x = marker_x[i]
          y = marker_y[i]
          marker_x.extend([x-1, x-1, x-1, x, x, x+1, x+1, x+1 #] )
                          
                          ,x-2, x-1, x , x+1, x+2,
                           x-2, x+2, x-2, x+2, x-2, x+2,
                           x-2, x-1, x, x+1, x+2   ])
                           
                           
                           #, x-3, x-2, x-1, x, x+1, x+2, x+3,
                           #x-3, x+3, x-3, x+3, x-3, x+3, x-3, x+3, x-3, x+3, 
                           #x-3, x-2, x-1, x, x+1, x+2, x+3
                           #])
          
          marker_y.extend([y+1,  y,  y-1,y+1,y-1,y+1, y,  y-1 #])
                           ,y+2,y+2,y+2,y+2,y+2,
                           y+1, y+1, y, y, y-1, y-1,
                           y-2, y-2, y-2, y-2, y-2  ])



                           #, y+3, y+3, y+3, y+3, y+3, y+3, y+3, 
                           #y+2, y+2,  y+1, y+1, y, y, y-1, y-1, y-2, y-2,
                           #y-3, y-3, y-3, y-3, y-3, y-3, y-3
                           #])





########################
    
        # 제로패딩하여 원하는 사이즈로 크롭해서 내보내는 함수
        def find_square (y, x, unit):
            square = im[find_min(y, unit):find_max(y, unit, length), find_min(x,unit):find_max(x, unit,width)]
            extend_x = size_hf*2 - find_max(x, unit,width) + find_min(x,unit)
            extend_y = size_hf*2 - find_max(y, unit,length) + find_min(y,unit)
            padding= np.lib.pad(square,((extend_y,0), (extend_x,0), (0,0)),'constant', constant_values=(255))
            return padding
            
        
        
        # 크롭 이미지 좌표 범위의 최솟값 결정
        def find_min (t, u):
            if t-u<=0:
                return 0
            elif t-u>=0:
                return t-u
            else:
                return None
        
        
            
        # 크롭 이미지 좌표 범위의 최댓값 결정
        def find_max (t, u, total):
            if t+u>=total:
                return total
            elif t+u<=total:
                return t+u
            else:
                return None
            

        #marker가 정가운데에 있는 crop 추가
        for i in range (0, len(marker_x)):
            X[last_num+i,:,:,:] = find_square(marker_y[i], marker_x[i], size_hf)
            Y[last_num+i,:] = [0,1]
            Z[last_num+i] = n
        
        # marker가 없는 crop 추가
        
        def blankspace(a,b):
            ok = 0
            for i in range(0, len(marker_x)):
                if (a-size_hf>marker_y[i] or a+size_hf<marker_y[i]) or (
                        (b-size_hf>marker_x[i] or b+size_hf<marker_x[i])):
                    ok+=1
            return ok
        
        import random
        random.seed(seed)
        negative = []
        AnB = []
        while len(negative) < len(marker_x):
            if len(negative) == len(marker_x):
                break
            
            A = random.randrange(min(marker_y)+1, max(marker_y)-1)
            B = random.randrange(min(marker_x)+1, max(marker_x)-1)
            
            if (blankspace(A,B) == len(marker_x)) and ((str(A)+"n"+str(B)) not in AnB):
                AnB.append(str(A)+"n"+str(B))
                negative.append(find_square(A,B,size_hf))
        
            
        # marker가 없는 crop, marker 개수만큼만 추가
        
        if len(negative) >= len(marker_x):
            for i in range (0,len(marker_x)):
                X[i+last_num+len(marker_x),:,:,:] = negative[i]
                Y[i+last_num+len(marker_x),:] = [1,0]
                Z[i+last_num+len(marker_x)] = n

        # negative 리스트 원소 개수가 marker 전체 수보다 적을 때 에러 방지
        elif len(negative) < len(marker_x):
            print('lack of ng')
        
        last_num += len(marker_x) *2
        
    else:
      continue
  
    
#%%
"""X,Y shffle"""
import random
random.seed(seed)

a = list(range(0,len(X)))
random.shuffle(a)

tr_X = np.zeros((len(X), size_hf*2, size_hf*2, 3)) * np.nan
tr_Y = np.zeros((len(X), 2)) * np.nan


# for i in range(0,len(X)):
#     tr_X[i,:,:,:] = X[a[i],:,:,:]
#     tr_Y[i,:] = Y[a[i],:]

# 개선-> np array는 여러 index를 동시에 처리할수 있어요.
tr_X[:,:,:,:] = X[a,:,:,:]
tr_Y[:,:] = Y[a,:]
    
#%%

""" model """

xs = X.shape[1:]
ys = Y.shape[1]


#%% keras setup
XS = xs
def model_setup():
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=xs))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(ys, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.summary()
    return model

model = model_setup()
initial_weightsave = 'saved_weight.h5'
model.save_weights(initial_weightsave)

#%%
"""test set 만들기"""
from PIL import Image

t_marker_x = dictionary[test_image_no]['marker_x']
t_marker_y = dictionary[test_image_no]['marker_y']
t_width = dictionary[test_image_no]['width']
t_length = dictionary[test_image_no]['length']
t_im = dictionary[test_image_no]['imread']
size = size_hf * 2



# roi 꼭짓점 _ roi의 끝 좌표가 가장자리에 있어 size_hf를 사방으로 확장할 수 없을 때를 대비

l_p = 0 ; u_p = 0 ; r_p = 0 ; d_p = 0 # 상하좌우 padding 얼마나 할지

left = int(min(t_marker_x))-size_hf
if left < 0:
    l_p = -1 * (left)
    left = 0
up = int(min(t_marker_y))-size_hf
if up < 0:
    u_p = -1 * (up)
    up = 0
right = int(max(t_marker_x))+size_hf
if right > t_width:
    r_p = right - t_width
    right = t_width
down = int(max(t_marker_y))+size_hf
if down > t_length:
    d_p = down - t_length
    


#%%
t_crops = (right - left-size+1) * (down-up-size+1) # padding된 roi 속 모든 crop 갯수
print('roi image 속 총 crop 수:', t_crops)

#%%
##setting
t_crops = (right - left-size+3) * (down-up-size+3) # padding된 roi 속 모든 crop 갯수
test_X = np.zeros((t_crops, size, size, 3)) * np.nan ## test_X, test_Y 세션다운 발생지점..
test_Y = np.zeros((t_crops, 2)) * np.nan

#%%
# roi 크롭
roi_im = t_im[up:down,left:right]

# roi cropped image를 padding (상하좌우 1 pixel씩)
p_roi_im = np.lib.pad(roi_im,((u_p,d_p), (l_p,r_p), (0,0)),'constant', constant_values=255) # 앞서 언급한 경우 방지
p_roi_len = p_roi_im.shape[0] #세로길이
p_roi_wid = p_roi_im.shape[1] #가로길이

print('original test image length:', t_length)
print('original test image width:', t_width)
print('padded roi image length:', p_roi_len)
print('padded roi image width:', p_roi_wid)
#%%
"""test X 구성하기"""
# padding 된 roi 이미지의 모든 crop 추가
test_X_list = []
for i in range(0,p_roi_len-size+3):
  for n in range (0, p_roi_wid-size+3):
      test_X_list.append(t_im[ i : i+size, n: n+size ])


# for i in range(0,t_crops):
#     test_X[i,:,:,:] = test_X_list [i]

# 개선 -> nested list (복잡하게 겹친 list)라고 하더라고, matrix로 만들수 있는 상태 (차원별로 배치가 정형화되어있으면)
# 바로 array로 전환이 가능합니다.
test_X_list = np.array(test_X_list)

#%%
# 헷갈리수 있으니 예시를 들면
# 정형화된 case
ms = [[1,2], [3,4]]
print(ms)
ms_array = np.array(ms)
print(ms_array)
print(ms_array.shape)

# 비 정형화된 case
ms = [[1,2], [3,]]
print(ms)
ms_array = np.array(ms)
print(ms_array)
print(ms_array.shape)
# 이 경우 array로 인식되기는 하지만 numpy 함수 사용에 제한이 있음.

#%%
"""test Y 구성하기"""
for i in range(0, t_crops):
  test_Y[i,:] = [1,0]

for i in range(0, len(t_marker_y)):
  test_Y[((right-left-size+1)*(t_marker_y[i]-size_hf-up+1))+(x-left-size_hf+1)] = [0,1]
  
# 이부분이 바로 이해가 안되네요. 다음주에 물어볼게요.
# X랑 Y를최대한 근접해서 작성해주세요.
  
#%%
""" 겹치는지 check"""
training_check = []
for nn in range(len(X)):
        training_check.append([np.mean(X[nn]), np.std(X[nn])])
for nn in range(len(test_X)):
  if [np.mean(test_X[nn]), np.std(test_X[nn])] in training_check:
    print('tr, te 겹침'); import sys; sys.exit()
    
#%%
#파일이름..
import datetime
now = datetime.datetime.now()
nowDateTime = now.strftime('%Y_%m_%d_%H_%M_%S')

#%%

#training
# model.fit(X, Y, epochs=5, verbose=1, validation_data = (test_X, test_Y))

# 개선-> ... X -> tr_X
# 직전에 항상 초기화하는게 안전해요

model = model_setup()
initial_weightsave = 'saved_weight.h5'
model.load_weights(initial_weightsave)
model.fit(tr_X, tr_Y, epochs=5, verbose=1, validation_data = (test_X, test_Y))

modelname = str(nowDateTime)+'saved_weight.hp'
model.save_weights(modelname)
test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2 )


#%%
import random
random.seed(seed)
"""test set shuffle """
te_X = np.zeros((len(test_X), size, size, 3)) * np.nan
te_Y = np.zeros((len(test_X), 2)) * np.nan

b = list(range(0, len(test_Y)))
random.shuffle(b)
    
    
for i in range(0, len(test_Y)):
    te_X[i,:,:,:] = test_X[b[i],:,:,:]
    te_Y[i,:] = test_Y[b[i],:]

#%%
# prediction
from tqdm import tqdm

pc = 100000
predictions = []
for i in tqdm(range(0,int(len(test_Y)/pc))):
    p = model.predict(test_X[i*pc:(i*pc)+pc,:,:,:])   #10->1
    p_frag = list(np.array(p[:,0] < 0.5, dtype=int))
    predictions.extend(p_frag)
#%%    
if int(len(test_Y)/pc) < len(test_Y/pc):
    rest = len(test_Y) - (pc * int(len(test_Y)/pc))
    p = model.predict(test_X[i:i+rest,:,:,:])
    p_frag = list(np.array(p[:,0] < 0.5, dtype=int))
    predictions.extend(p_frag)
    
print(len(predictions))
                             
# if p[0][0] < 0.5:
#   predictions.append(0)
# else:
#   predictions.append(1) 
#print('len prediction:', len(predictions),', len test Y:', len(test_Y))
#%%
# 임의 acc

#t_crops = 10000 ########################################################################### 이 줄 삭제
pos = 0
for i in range (0,t_crops): 
   if test_Y[i,:][0] == 0:
     pos +=1

neg = len(predictions)- pos


tp = 0
for i in range(0,t_crops):
  if (int(test_Y[i,:][0])==0) and (int(predictions[i])==0):
    tp += 1
sen = (tp / pos) * 100
print("truepos="+str(sen)+"%")


tn=0
for i in range(0,t_crops):
  if (int(test_Y[i,:][0])==1) and (int(predictions[i])==1):
    tn += 1
sp = (tn / neg) * 100
print("trueneg="+str(sp)+"%")



#%%
""" 예측 이미지 """
import matplotlib.pylab as plt
predict_X = []
predict_Y = []

for i in range (len(test_Y)):
  if predictions[i] == 0:
    predict_X.append(i - int(i/(right-left-size+1)) + size_hf + left - 1)
    predict_Y.append(int(i/(right-left-size+1)))

import matplotlib.pylab as plt

plt.xlabel('predict_X')
plt.ylabel('predict_Y')
plt.show()

#%%
"""정답 이미지"""
answer_X = []
answer_Y = []

for i in range (len(test_Y)):
  if test_Y[i][0] == 0:
    answer_X.append(i - int(i/(right-left-size+1)) + size_hf + left - 1)
    answer_Y.append(int(i/(right-left-size+1)))

print(answer_X)

import matplotlib.pylab as plt
plt.xlabel('answer_X')
plt.ylabel('answer_Y')
plt.show()
#%%
"""원본이미지 -  original > roi"""
from PIL import Image
import matplotlib.image as img
import matplotlib.pyplot as pp
pp.imshow(p_roi_im)
