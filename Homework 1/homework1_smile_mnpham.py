import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import time

def fPC(y, yhat):
    """This function calculates accuracy
    @param y: 1D array: ground truth
    @param yhat: 1D array: predicted
    return double: accuracy
    """
    return np.mean(y == yhat)

def measureTime(start, end):
    return (end - start)

def drawFeatures(predictors, testingFaces):
    im = testingFaces[0,:,:]
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    # Show r1,c1
    for location in predictors:
        r1, c1, r2, c2  = location
        rect = patches.Rectangle((c1,r1),1,1,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2,r2),1,1,linewidth=2,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    # Display the merged result
    plt.show()

def measureAccuracyOfPredictors (predictors, X, y):
    """Given a list of predictors, this function measure accuracy
    @param predictors: 2D list: list of features
    @param X: 2D array: list of photos
    @param y: 1D array: ground truth
    return double: accuracy
    """
    yhat = np.zeros(y.shape)
    for feature in predictors:
        row1 = feature[0]
        col1 = feature[1]
        row2 = feature[2]
        col2 = feature[3]
        pixel_value1 = X[:,row1,col1]
        pixel_value2 = X[:,row2,col2]
        y_temp = pixel_value1 - pixel_value2
        y_temp[y_temp > 0] = 1 #smile
        y_temp[y_temp <= 0] = 0 #not smile
        yhat = yhat + y_temp
    yhat = yhat/len(predictors)
    yhat[yhat > 0.5] = 1
    yhat[yhat <= 0.5] = 0
    return fPC(y,yhat)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    n = [400, 800, 1200, 1600, 2000]

    result = {}

    for training_num in n:
        training_X = trainingFaces[:training_num]
        training_y = trainingLabels[:training_num]
        predictors = []
        count = 0
        
        #start timer
        start = time.time()
        print("Size: {}".format(training_num))
        while(len(predictors) < 5):
            accuracy = 0
            feature = []
            
            for row1 in range(24):
                for col1 in range(24):
                    for row2 in range(24):
                        for col2 in range(24):
                            if [row1,col1,row2,col2] in feature:
                                print("Works!")
                                continue
                            feature_temp = [row1,col1,row2,col2]
                            predictors_temp = predictors
                            predictors_temp = predictors_temp + [feature_temp]
                            accuracy_temp = measureAccuracyOfPredictors(predictors_temp, training_X, training_y)
                            if(accuracy_temp > accuracy):
                                feature = feature_temp #update best feature
                                accuracy = accuracy_temp #update best accuracy                  
            predictors.append(feature)
            print("Chosen predictor: ", feature)
            result[training_num] = predictors
        #end timer
        end = time.time()
        print("Time: {} seconds \n".format(measureTime(start, end)))
        
    show = False
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((c1,r1),1,1,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2,r2),1,1,linewidth=2,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()
        
    return result

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    faces_train, label_train = loadData("train")
    faces_test, label_test = loadData("test")
    result = stepwiseRegression(faces_train, label_train, faces_test, label_test)

    #Print out accuracy
    s = [400, 800, 1200, 1600, 2000]
    for n in s:
        print("Size: {}".format(n))
        print("Training Accuracy: {}".format(measureAccuracyOfPredictors(result[n],faces_train[:n,],label_train[:n])*100))
        print("Testing Accuracy: {} \n".format(measureAccuracyOfPredictors(result[n],faces_test,label_test)*100))