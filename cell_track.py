import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

path_list = []
folder_root = './Sequences'
folder_paths_original = ['/01','/02','/03','/04']
for folder_path in folder_paths_original:
    temp_list = []
    with os.scandir(folder_root+folder_path) as files:
        for file in files:
            temp_list.append(folder_root+folder_path+'/'+file.name)
    path_list.append(temp_list)

    def initIndexColorsTraj():
        a = list(range(0,255,10))
        a = a[1:]
        colorList = []
        indexTrajs = []
        for r in a:
            for g in a:
                for b in a:
                    colorList.append((r,g,b))
        random.shuffle(colorList)
        indexColor = []
        index = 1
        for item in colorList:
            indexColor.append([index,item,[],[], 1])
            index += 1
        return np.array(indexColor, dtype=object)

    def getSeg(img_path):
        img = cv2.imread(img_path)
        img_gray = cv2.imread(img_path,-1)
        # Image Normalization
        img_norm = cv2.normalize(img_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # Pre-processing:
        img_norm = img_norm.astype('uint8')
        # binary threshold
        # otsu threshold
        # Adaptive Threshold
        img_norm = cv2.blur(img_norm, (11, 11))
        thresh = cv2.adaptiveThreshold(img_norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 0)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.45*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        # watershed
        markers = cv2.watershed(img,markers)
        bi_markers = markers.copy()
        bi_markers[bi_markers>1] = 255
        bi_markers[bi_markers<=1] = 0
        img_bi_markers = bi_markers.copy().astype('uint8')
        contours, hierarchy = cv2.findContours(img_bi_markers,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        area = []
        contours_sure = []
        for i in contours:
            if cv2.contourArea(i) > 80:
                contours_sure.append(i)
        return contours_sure

    def drawContoursIndex_init(contours, img_path, indexColorList, video):
        img = cv2.imread(img_path, -1)
        img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_norm = img_norm.astype('uint8')
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
        img_withContours = None
        img_contoursIndex = None
        # Drop unmatched Cells
        indexColorList = indexColorList[indexColorList[:,4]==1]
        for i in range(len(contours)):
            indexColorList = indexColorList
            color = indexColorList[:,1][i]
            img_withContours = cv2.drawContours(img_rgb, [contours[i]], -1, color, 1)
            index = indexColorList[:,0][i]
            mu = cv2.moments(contours[i])
            mc = (int(mu['m10']/mu['m00']), int(mu['m01']/mu['m00']))
            img_contoursIndex = cv2.putText(img_withContours, str(index), mc, 1, 1, (255,255,255), 1)
            indexColorList[:,2][i].append(mc)
        countCell = 0
        avgSizeCell = 0
        avgDisp = 0
        numDividing = 0
        img_contoursIndex = cv2.putText(img_contoursIndex, 'Cell Count: ', (0,12), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, str(countCell), (6,24), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, 'Avg. Size: ', (0,36), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, str(avgSizeCell), (6,48), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, 'Avg. Displacement: ', (0,60), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, str(avgDisp), (6,72), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, 'Dividing Cells: ', (0,84), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, str(numDividing), (6,96), 1, 1, (255,255,255), 1)
        # Write Video
        video.write(img_contoursIndex)

    def drawContoursIndex(contours, img_path, indexColorList, prevCont, newContours, prevContLink, video, ghostRectList):
        img = cv2.imread(img_path, -1)
        img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img_norm = img_norm.astype('uint8')
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
        img_withContours = None
        img_contoursIndex = None
        # Drop unmatched Cells
        indexColorList = indexColorList[indexColorList[:,4]==1]
        # Compute Displacement
        totalDisp = 0
        for idx in range(len(prevContLink)):
            if prevContLink[idx] != -1:
                preMu = cv2.moments(prevCont[idx])
                preMc = (int(preMu['m10']/preMu['m00']), int(preMu['m01']/preMu['m00']))
                newMu = cv2.moments(newContours[prevContLink[idx]])
                newMc = (int(newMu['m10']/newMu['m00']), int(newMu['m01']/newMu['m00']))
                totalDisp += np.sqrt(np.add(np.square(np.subtract(preMc[0],newMc[0])), np.square(np.subtract(preMc[1],newMc[1]))))
        # Compute Cells Area and Number of Dividing Cells
        totalArea = 0
        divideCounter = 0
        for i in range(len(contours)):
            (x, y, w, h) = cv2.boundingRect(contours[i])
            if (x>0 and y>0 and (x+w)<img.shape[1] and (y+h)<img.shape[0]):
                totalArea += cv2.contourArea(contours[i])
            indexColorList = indexColorList
            color = indexColorList[:,1][i]
            img_withContours = cv2.drawContours(img_rgb, [contours[i]], -1, color, 1)
            # Display Cell Index
            index = indexColorList[:,0][i]
            mu = cv2.moments(contours[i])
            mc = (int(mu['m10']/mu['m00']), int(mu['m01']/mu['m00']))
            img_contoursIndex = cv2.putText(img_withContours, str(index), mc, 1, 1, (255,255,255), 1)
            # Record previous centroid of cells
            indexColorList[:,2][i].append(mc)
            for pointIdx in range(len(indexColorList[:,2][i])-1):
                cv2.line(img_contoursIndex, indexColorList[:,2][i][pointIdx], indexColorList[:,2][i][pointIdx+1], color, 1)
            # Check for dividing
            (centerX,centerY),(width,height),rotation = cv2.fitEllipse(contours[i])
            majMin = height/width
            if majMin<1:
                majMin = 1/majMin
            if majMin>=2:
                indexColorList[:,3][i].clear()
                indexColorList[:,3][i].append(contours[i])
            else:
                indexColorList[:,3][i].clear()
            if len(indexColorList[:,3][i]) == 1:
                (x, y, w, h) = cv2.boundingRect(indexColorList[:,3][i][0])
                img_contoursIndex = cv2.rectangle(img_contoursIndex,(x,y),(x+w,y+h),(0,0,255),1)
                divideCounter += 1
        # Draw alert after dividing
        for ghostRect in ghostRectList:
            (x, y, w, h) = cv2.boundingRect(ghostRect[0])
            img_contoursIndex = cv2.rectangle(img_contoursIndex,(x,y),(x+w,y+h),(0,0,255),1)
            divideCounter += 1
        # Ghost Alert Only Display Once
        ghostRectList.clear()
        # Display Stats
        countCell = len(contours)
        avgSizeCell = totalArea/countCell
        avgDisp = totalDisp/countCell
        numDividing = divideCounter
        img_contoursIndex = cv2.putText(img_contoursIndex, 'Cell Count: ', (0,12), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, str(countCell), (6,24), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, 'Avg. Size: ', (0,36), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, str(round(avgSizeCell,4)), (6,48), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, 'Avg. Displacement: ', (0,60), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, str(round(avgDisp,4)), (6,72), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, 'Dividing Cells: ', (0,84), 1, 1, (255,255,255), 1)
        img_contoursIndex = cv2.putText(img_contoursIndex, str(numDividing), (6,96), 1, 1, (255,255,255), 1)
        # Write Video
        plt.imshow(img_contoursIndex)
        plt.title(img_path)
        plt.show()
        video.write(img_contoursIndex)

        def main():
            videoCounter = 1
            for paths in path_list:
                print('Processing Folder ' + str(videoCounter))
                # Video
                videoName = str(videoCounter)+'_output.avi'
                videoCounter += 1
                video = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('I', '4', '2', '0'), 2, (1100, 700))
                # get RGB color space
                indexColorList = initIndexColorsTraj()
                ghostRectList = []
                counter = 0
                currentCont = None
                for path in paths:
                    newContours = getSeg(path)
                    # index > 1
                    if currentCont != None:
                        # find contours distance < M
                        linkList = []
                        for i in range(len(currentCont)):
                            contPrev = currentCont[i]
                            muP = cv2.moments(contPrev,False)
                            mcP = (muP['m10']/muP['m00'], muP['m01']/muP['m00'])
                            dtype = [('Index', int), ('distance', float), ('area', float), ('score', float)]
                            prevNewDist = np.array([], dtype=dtype)
                            for j in range(len(newContours)):
                                contNew = newContours[j]
                                muN = cv2.moments(contNew,False)
                                mcN = (muN['m10']/muN['m00'], muN['m01']/muN['m00'])
                                # solve dist == 0
                                dist = np.add(np.square(np.subtract(mcP[0],mcN[0])), np.square(np.subtract(mcP[1],mcN[1])))+0.0001
                                if dist < 60.0:
                                    area = cv2.contourArea(contNew)
                                    prevNewDist = np.append(prevNewDist, np.array([(j,dist,area,0)],dtype=dtype))
                            if len(prevNewDist)!=0:
                                prevNewDist['distance'] = 1-prevNewDist['distance']/np.max(prevNewDist['distance'])
                                areaP = cv2.contourArea(contPrev)
                                prevNewDist['area'] = 1-np.abs(prevNewDist['area']-areaP)/areaP
                                prevNewDist['score'] = prevNewDist['distance']+prevNewDist['area']*0.4
                                prevNewDist['score'] = -prevNewDist['score']
                                prevNewDist = np.sort(prevNewDist, order='score')
                            linkList.append(prevNewDist['Index'].tolist())
                        # Get Optimal old-new link
                        newContSet = set(range(len(newContours)))
                        newIdxList = []
                        for nextIdxs in linkList:
                            if len(nextIdxs)==0:
                                newIdxList.append(-1)
                            else:
                                for m in range(len(nextIdxs)):
                                    nextidx = nextIdxs[m]
                                    if nextidx in newContSet:
                                        newIdxList.append(nextidx)
                                        newContSet.remove(nextidx)
                                        break
                                    if m>=(len(nextIdxs)-1):
                                        newIdxList.append(-1)
                        # newIdxList: Optimal old-new link [int/-1]
                        # newContSet: Cells in new image that did not have previous cell [int]
                        # prevCont: Previous contours with next cell
                        # Delete row in index-color list
                        for idx in range(len(newIdxList)):
                            if newIdxList[idx] == -1:
                                indexColorList[idx][4] = 0
                                # Add ghost Rect Dividing Hints
                                if len(indexColorList[idx][3]) == 1:
                                    ghostRectList.append(indexColorList[idx][3])
                        # Drop unmatched Cells
                        indexColorList = indexColorList[indexColorList[:,4]==1]

                        prevContLink = newIdxList.copy()
                        # Remove -1 value in newIdxList
                        newIdxList = list(filter(lambda x:x>=0, newIdxList))

                        newContourList = []
                        for idx in newIdxList:
                            newContourList.append(newContours[idx])
                        for idx in newContSet:
                            newContourList.append(newContours[idx])
                        # Update and record prevCont
                        prevCont = currentCont.copy()
                        currentCont = newContourList
                        drawContoursIndex(currentCont, path, indexColorList, prevCont, newContours, prevContLink, video, ghostRectList)
                    else:
                        currentCont = newContours
                        drawContoursIndex_init(currentCont, path, indexColorList, video)
                video.release()
                cv2.destroyAllWindows()




def main():
    videoCounter = 1
    for paths in path_list:
        print('Processing Folder ' + str(videoCounter))
        # Video
        videoName = str(videoCounter) + '_output.avi'
        videoCounter += 1
        video = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('I', '4', '2', '0'), 2, (1100, 700))
        # get RGB color space
        indexColorList = initIndexColorsTraj()
        ghostRectList = []
        counter = 0
        currentCont = None
        for path in paths:
            newContours = getSeg(path)
            # index > 1
            if currentCont != None:
                # find contours distance < M
                linkList = []
                for i in range(len(currentCont)):
                    contPrev = currentCont[i]
                    muP = cv2.moments(contPrev, False)
                    mcP = (muP['m10'] / muP['m00'], muP['m01'] / muP['m00'])
                    dtype = [('Index', int), ('distance', float), ('area', float), ('score', float)]
                    prevNewDist = np.array([], dtype=dtype)
                    for j in range(len(newContours)):
                        contNew = newContours[j]
                        muN = cv2.moments(contNew, False)
                        mcN = (muN['m10'] / muN['m00'], muN['m01'] / muN['m00'])
                        # solve dist == 0
                        dist = np.add(np.square(np.subtract(mcP[0], mcN[0])),
                                        np.square(np.subtract(mcP[1], mcN[1]))) + 0.0001
                        if dist < 60.0:
                            area = cv2.contourArea(contNew)
                            prevNewDist = np.append(prevNewDist, np.array([(j, dist, area, 0)], dtype=dtype))
                    if len(prevNewDist) != 0:
                        prevNewDist['distance'] = 1 - prevNewDist['distance'] / np.max(prevNewDist['distance'])
                        areaP = cv2.contourArea(contPrev)
                        prevNewDist['area'] = 1 - np.abs(prevNewDist['area'] - areaP) / areaP
                        prevNewDist['score'] = prevNewDist['distance'] + prevNewDist['area'] * 0.4
                        prevNewDist['score'] = -prevNewDist['score']
                        prevNewDist = np.sort(prevNewDist, order='score')
                    linkList.append(prevNewDist['Index'].tolist())
                # Get Optimal old-new link
                newContSet = set(range(len(newContours)))
                newIdxList = []
                for nextIdxs in linkList:
                    if len(nextIdxs) == 0:
                        newIdxList.append(-1)
                    else:
                        for m in range(len(nextIdxs)):
                            nextidx = nextIdxs[m]
                            if nextidx in newContSet:
                                newIdxList.append(nextidx)
                                newContSet.remove(nextidx)
                                break
                            if m >= (len(nextIdxs) - 1):
                                newIdxList.append(-1)
                # newIdxList: Optimal old-new link [int/-1]
                # newContSet: Cells in new image that did not have previous cell [int]
                # prevCont: Previous contours with next cell
                # Delete row in index-color list
                for idx in range(len(newIdxList)):
                    if newIdxList[idx] == -1:
                        indexColorList[idx][4] = 0
                        # Add ghost Rect Dividing Hints
                        if len(indexColorList[idx][3]) == 1:
                            ghostRectList.append(indexColorList[idx][3])
                # Drop unmatched Cells
                indexColorList = indexColorList[indexColorList[:, 4] == 1]

                prevContLink = newIdxList.copy()
                # Remove -1 value in newIdxList
                newIdxList = list(filter(lambda x: x >= 0, newIdxList))

                newContourList = []
                for idx in newIdxList:
                    newContourList.append(newContours[idx])
                for idx in newContSet:
                    newContourList.append(newContours[idx])
                # Update and record prevCont
                prevCont = currentCont.copy()
                currentCont = newContourList
                drawContoursIndex(currentCont, path, indexColorList, prevCont, newContours, prevContLink, video,
                                      ghostRectList)
            else:
                currentCont = newContours
                drawContoursIndex_init(currentCont, path, indexColorList, video)
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
