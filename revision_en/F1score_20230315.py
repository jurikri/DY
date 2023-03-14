#%%
def get_F1(threshold=None, contour_thr=None,\
           yhat_save=None, z_save=None, t_im=None, positive_indexs=None, polygon=None):

    import numpy as np
    # import cv2
    
    height = t_im.shape[0]
    width = t_im.shape[1]
    
    noback_img = np.zeros((height, width, 3), np.uint8)
    noback_img[:]= [255,255,0]
    
    # test_img = np.zeros((height, width))

    vix = np.where(np.array(yhat_save) > threshold)[0]
    for i in vix:
        row = z_save[i][0]
        col = z_save[i][1]
        noback_img[row,col] = [0,0,255]
        # test_img[row,col] = 1
        
    # plt.imshow(test_img)

    img_color = noback_img.copy()
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    pink = [] # predicted cell 중앙 좌표
    los = [] # size filter 에서 살아남는 contours 
    
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        if M['m00'] >= contour_thr and  M['m00'] <= 500: 
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            code = Point(cx,cy)
            if not(polygon is None):
                if code.within(polygon):
                    pink.append((cx,cy))
                    los.append(contours[i])
            else:
                pink.append((cx,cy))
                los.append(contours[i])
                
    def dot_expand(height=None, width=None, dots=None):
        white2 = np.zeros((height,width,3))*np.nan
        white2[:,:] = [255,255,255]
        boxsize = 10
        for z in range(len(dots)):
            row = dots[z][1]
            col = dots[z][0]
            white2[np.max([row-boxsize, 0]) : np.min([row+boxsize, height]), \
                   np.max([col-boxsize, 0]) : np.min([col+boxsize, width])] = 0
        return white2

    dots = list(pink)
    white2 = dot_expand(height=height, width=width, dots=dots)
    predict_area = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        predict_area.append((w[1][i],w[0][i]))
    # plt.imshow(white2)

    co = []
    for i in range(len(positive_indexs)):
        row = positive_indexs[i][0]
        col = positive_indexs[i][1]
        code = Point(col, row)
        if not(polygon is None):
            if code.within(polygon):
                co.append((int(col),int(row)))
        else: co.append((int(col),int(row)))
        
    dots = list(co)
    white2 = dot_expand(height=height, width=width, dots=dots)
    cell_positive_area = []
    w = np.where(white2[:,:,0]==0)
    for i in range(len(w[0])):
        cell_positive_area.append((w[1][i],w[0][i]))
        
    if True:
            # TN
        dots = list(co)
        ansbox_p = dot_expand(height=height, width=width, dots=dots)
        ansbox = np.array(ansbox_p[:,:,0])
        predbox = np.array(white2[:,:,0])
        predbox2 = np.zeros(predbox.shape)
        predbox2[np.where(predbox==0)] = 255
        
        # plt.imshow(predbox2)
        # plt.imshow(ansbox)
        # plt.imshow(ansbox - predbox2)
        ansbox3 = ((ansbox - predbox2) == 255)
        # plt.imshow(ansbox3)
        TN = np.sum(ansbox3)
    
        Cell_n = len(co)
        Predict_n = len(pink)
        
        FN = len(list(set(co) - set(predict_area)))
        TP = Cell_n - FN
        
        FP = len(list(set(pink) - set(cell_positive_area)))
        
        if np.min([FN, TP, FP]) < 0:
            print(threshold, contour_thr)
            import sys; sys.exit()
    
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2 * precision * recall / (precision + recall)
        # acc = (TP + TN) / (TP + TN + FP + FN)
    except: F1_score = 0; #  acc = 0
    
    # plt.imshow(predbox2)
    # plt.imshow(ansbox)
    
    msdict = {'los': los, 'co': co, 'tp': TP, 'fp': FP, 'fn': FN, \
              'predbox2': predbox2, 'ansbox': ansbox}
    
    return F1_score, msdict

#%% 
