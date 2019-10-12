import os, cv2, numpy as np

def get_path_list(root_path):
    '''
        To get a list of path directories from root xpath

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    paths = os.listdir(root_path)
    return paths

def get_class_names(root_path, train_names):
    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''
    fileId = []
    path = []
    for id, name in enumerate(train_names):
        images = os.listdir(root_path+"/"+name)
        for image in images:
            path.append(root_path+"/"+name+"/"+image)
            fileId.append(id)
    
    return path, fileId

def get_train_images_data(image_path_list):
    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''
    loadedImg = []
    for imageP in image_path_list:
        loadedImg.append(cv2.imread(imageP))
    
    return loadedImg

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    face_loc = []
    face_crop = []
    face_id = []

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for idx, image in enumerate(image_list):
        face_gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_Faces = face_cascade.detectMultiScale(face_gray, scaleFactor=1.3, minNeighbors=3)
        if len(detected_Faces) != 1: 
            continue
        else:
            for face in detected_Faces:
                x, y, w, h = face
                croppedFace = face_gray[y:y+h, x:x+w]
                face_loc.append(face)
                face_crop.append(croppedFace)
                if(image_classes_list != None) :
                    face_id.append(image_classes_list[idx])

    
    return face_crop, face_loc, face_id

def train(train_face_grays, image_classes_list):
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''
    
    object_train = cv2.face.LBPHFaceRecognizer_create()
    object_train.train(train_face_grays, np.array(image_classes_list))  
    return object_train

def get_test_images_data(test_root_path, image_path_list):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''
    loadedImg = []
    for imageP in image_path_list:
        loadedImg.append(cv2.imread(test_root_path+"/"+imageP))
    
    return loadedImg

def predict(classifier, test_faces_gray):
    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    resList = []
    for image in test_faces_gray:
        res, _ = classifier.predict(image)
        resList.append(res)
    
   
    return resList

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    '''

    for idx in range(len(predict_results)):
        x, y, w, h = test_faces_rects[idx]
        img_name = train_names[predict_results[idx]]
        img_name = img_name.split("_")
        img_name = " ".join(img_name)
        cv2.rectangle(test_image_list[idx], (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.putText(test_image_list[idx], img_name, (x, y-15), 0, 0.4, (0, 255, 0), 1)
    
    return test_image_list
        

def combine_results(predicted_test_image_list):
    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''
    combination = np.hstack(tuple(predicted_test_image_list))
    return combination

def show_result(image):
    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"#
    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)#
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)#
    train_image_list = get_train_images_data(image_path_list)#
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)#
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)#
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)#
    final_image_result = combine_results(predicted_test_image_list)#
    show_result(final_image_result)#