import cv2
import dlib
import numpy

#------ Function Declaration --------------------------------------------------------------------------------------------------------------------------------------------------------------
  

def r_eye(a,b):
    xc = []
    yc = []
    for i in range(36, 42):
        xc.append(a[i])
        yc.append(b[i])
    return xc,yc

def l_eye(a,b):
    xc = []
    yc = []
    for i in range(42, 48):
        xc.append(a[i])
        yc.append(b[i])
    return xc,yc

def r_eyebrow(a,b):
    xc = []
    yc = []
    for i in range(17, 22):
        xc.append(a[i])
        yc.append(b[i])
    return xc,yc

def l_eyebrow(a,b):
    xc = []
    yc = []
    for i in range(22, 27):
        xc.append(a[i])
        yc.append(b[i])
    return xc,yc

def eye_distance(img):
    x=[]    #Local Vairable Declaration
    y=[]

    #Image Processing for analysis
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    #cv2.imshow(winname="sample", mat=gray)

    #Calling Face Feature Detector Function
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")       #Imports face feature coordinate extractor model

    #Using Face Feature Detector Function
    faces = detector(gray)

    for face in faces:

        # Look for the landmarks
        for n in range(0, 68):
            landmarks = predictor(image=gray, box=face)
            x.append(landmarks.part(n).x)
            y.append(landmarks.part(n).y)

    #Function Calling to sepearte feature coodrinates
    r_eyex,r_eyey = r_eye(x,y)
    l_eyex,l_eyey = l_eye(x,y)
    r_eyebrowx,r_eyebrowy = r_eyebrow(x,y)
    l_eyebrowx,l_eyebrowy = l_eyebrow(x,y)

    #Calculating Euclidean Distance
    dist_rar = []   #Right Eye side distances array
    dist_lar = []   #Left Eye side distances array

    for i in range (0,2):
        point1 = numpy.array((r_eyebrowx[3],r_eyebrowy[3]))     #Right Eyebrow center coordinate
        point2 = numpy.array((r_eyex[3+i],r_eyey[3+i]))         #Right Eye center coordinates
        dist_rar.append(numpy.linalg.norm(point1 - point2))     #Right Eye Euclidean Distances
        point3 = numpy.array((l_eyebrowx[3],l_eyebrowy[3]))     #Left Eyebrow center coordinate
        point4 = numpy.array((l_eyex[3+i],l_eyey[3+i]))         #Left Eye center coordinates
        dist_lar.append(numpy.linalg.norm(point3 - point4))     #Left Eye Euclidean Distances

    dist_r = numpy.mean(dist_rar)
    dist_l = numpy.mean(dist_lar)
    
    return dist_r,dist_l

def eye_size(img):
    x=[]    #Local Vairable Declaration
    y=[]

    #Image Processing for analysis
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    #cv2.imshow(winname="sample", mat=gray)

    #Calling Face Feature Detector Function
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")       #Imports face feature coordinate extractor model

    #Using Face Feature Detector Function
    faces = detector(gray)

    for face in faces:

        # Look for the landmarks
        for n in range(0, 68):
            landmarks = predictor(image=gray, box=face)
            x.append(landmarks.part(n).x)
            y.append(landmarks.part(n).y)

    #Function Calling to sepearte feature coodrinates
    r_eyex,r_eyey = r_eye(x,y)
    l_eyex,l_eyey = l_eye(x,y)


    #Calculating Euclidean Distance
    dist_rar = []   #Right Eye side distances array
    dist_lar = []   #Left Eye side distances array
    eyept = [2,3]

    for i in range (0,2):
        point1 = numpy.array((r_eyex[eyept[i]],r_eyey[eyept[i]]))     #Right Eye Top coordinate
        temp = len(r_eyex)
        point2 = numpy.array((r_eyex[temp-1-i],r_eyey[temp-1-i]))         #Right Eye Bottom coordinates
        dist_rar.append(numpy.linalg.norm(point1 - point2))     #Right Eye Euclidean Distances
        point3 = numpy.array((l_eyex[eyept[i]],l_eyey[eyept[i]]))     #Left Eye Top coordinate
        point4 = numpy.array((l_eyex[temp-1-i],l_eyey[temp-1-i]))         #Left Eye Bottom coordinates
        dist_lar.append(numpy.linalg.norm(point3 - point4))     #Left Eye Euclidean Distances

    dist_r = numpy.mean(dist_rar)
    dist_l = numpy.mean(dist_lar)
    
    return dist_r,dist_l

def eye_aspect_ratio(img):
    x=[]    #Local Vairable Declaration
    y=[]

    #Image Processing for analysis
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    #cv2.imshow(winname="sample", mat=gray)

    #Calling Face Feature Detector Function
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")       #Imports face feature coordinate extractor model

    #Using Face Feature Detector Function
    faces = detector(gray)

    for face in faces:

        # Look for the landmarks
        for n in range(0, 68):
            landmarks = predictor(image=gray, box=face)
            x.append(landmarks.part(n).x)
            y.append(landmarks.part(n).y)

    #Function Calling to sepearte feature coodrinates
    r_eyex,r_eyey = r_eye(x,y)
    l_eyex,l_eyey = l_eye(x,y)


    #Calculating Euclidean Distance
    dist_rar = []   #Right Eye side distances array
    dist_lar = []   #Left Eye side distances array
    eyept = [2,3]

    for i in range (0,2):
        point1 = numpy.array((r_eyex[eyept[i]],r_eyey[eyept[i]]))     #Right Eye Top coordinate
        temp = len(r_eyex)
        point2 = numpy.array((r_eyex[temp-1-i],r_eyey[temp-1-i]))         #Right Eye Bottom coordinates
        dist_rar.append(numpy.linalg.norm(point1 - point2))     #Right Eye Euclidean Distances
        point3 = numpy.array((l_eyex[eyept[i]],l_eyey[eyept[i]]))     #Left Eye Top coordinate
        point4 = numpy.array((l_eyex[temp-1-i],l_eyey[temp-1-i]))         #Left Eye Bottom coordinates
        dist_lar.append(numpy.linalg.norm(point3 - point4))     #Left Eye Euclidean Distances

    dist_r = numpy.mean(dist_rar)
    dist_l = numpy.mean(dist_lar)
    
    point5 = numpy.array((r_eyex[0],r_eyey[0]))
    point6 = numpy.array((r_eyex[3],r_eyey[3]))
    dist_rh = numpy.linalg.norm(point5 - point6) 
    point7 = numpy.array((l_eyex[0],l_eyey[0]))
    point8 = numpy.array((l_eyex[3],l_eyey[3]))
    dist_lh = numpy.linalg.norm(point7 - point8)

    aspect_ratio_r = dist_r/dist_rh
    aspect_ratio_l = dist_l/dist_lh

    aspect_ratio = numpy.mean([aspect_ratio_r,aspect_ratio_l])
    
    return aspect_ratio

def image_list():
    #Creating a list withh all the image filenames
    import os
    imgs = []
    data = []
    files = os.listdir()
    for file in files:
         if file.endswith(('_0.jpg', '_0.jpeg','_1.jpg', '_1.jpeg')):
             imgs.append(file)
             if file.endswith(('_0.jpg', '_0.jpeg')):
                 data.append(0)
             else:
                 data.append(1)
    return imgs, data

def image_test():
    #Creating a list withh all the image filenames
    import os
    imgs = []
    files = os.listdir()
    for file in files:
         if file.startswith(('test')):
             imgs.append(file)
    return imgs

def image_list_lab():
    #Creating a list withh all the image filenames
    import os
    imgs_0 = []
    imgs_1 = []
    files = os.listdir()
    for file in files:
         if file.endswith(('_0.jpg', '_0.jpeg')):
             imgs_0.append(file)
         if file.endswith(('_1.jpg', '_1.jpeg')):
             imgs_1.append(file)
    return imgs_0,imgs_1


#------ Main Code--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

#Variable Declaration
eye_eyebrow_dist=[]
eye_dist = []
eye_aspectratio = []
Ydata = []

#Procuring Imaages
images, Ydata = image_list()
size = len(images)

#Image Reading
for i in range (0,size):
    img = cv2.imread(images[i])
    print(images[i])

    #Image resizing
    scale = 35
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #Calculate Eye distances
    reb_e,leb_e = eye_distance(image)
    dist_eb_e = [reb_e,leb_e]

    #Calculate Eye Size
    reye,leye = eye_size(image)
    dist_eye = [reye,leye]

    #Calculate Eye Aspect Ratio
    aspectratio = eye_aspect_ratio(image)
    
    #Find mean distance out of two eyes
    max_dist_eb_e = numpy.max(dist_eb_e)
    max_dist_eye = numpy.max(dist_eye)

    #List of all Features
    eye_eyebrow_dist.append(max_dist_eb_e)
    eye_dist.append(max_dist_eye)
    eye_aspectratio.append(aspectratio)




#Using K-means Clustering to find the two clusters
"""
from sklearn.cluster import KMeans
Xdata = list(zip(eye_eyebrow_dist, eye_dist,eye_aspectratio))

kmeans = KMeans(n_clusters=2)
kmeans.fit(Xdata)
k_means_predicted = kmeans.predict(Xdata)
identified_clusters = kmeans.cluster_centers_
print(identified_clusters)


#Saving Centers into a Text file for later testing purpose

file = open("centroids.txt", "w+")
content = str(identified_clusters)
file.write(content)
file.close()


numpy.save("centroids",identified_clusters)
"""




#Testing Phase
test_eye_eyebrow_dist = []
test_eye_dist = []
test_eye_aspectratio = []

test_data = image_test()
test_size = len(test_data)

#Image Reading
for i in range (0,test_size):
    img = cv2.imread(test_data[i])
    print(test_data[i])

    #Image resizing
    scale = 35
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #Calculate Eye distances
    reb_e,leb_e = eye_distance(image)
    dist_eb_e = [reb_e,leb_e]

    #Calculate Eye Size
    reye,leye = eye_size(image)
    dist_eye = [reye,leye]

    #Calculate Eye Aspect Ratio
    aspectratio = eye_aspect_ratio(image)
    
    #Find mean distance out of two eyes
    max_dist_eb_e = numpy.max(dist_eb_e)
    max_dist_eye = numpy.max(dist_eye)

    #List of all Features
    test_eye_eyebrow_dist.append(max_dist_eb_e)
    test_eye_dist.append(max_dist_eye)
    test_eye_aspectratio.append(aspectratio)


#Using K-Nearest Neighbour to predict the incoming image
from sklearn.neighbors import KNeighborsClassifier
imagedata = list(zip(eye_eyebrow_dist, eye_dist, eye_aspectratio))
classes = Ydata
testdata = list(zip(test_eye_eyebrow_dist, test_eye_dist, test_eye_aspectratio))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(imagedata, classes)
prediction = knn.predict(testdata)
prediction_size = len(prediction)

for i in range (0, prediction_size):
    if ( prediction[i] == 0):
        print(test_data[i]," is not drowsy")
    else:
        print(test_data[i]," is drowsy")    





      









