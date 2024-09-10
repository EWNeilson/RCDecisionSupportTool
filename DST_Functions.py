
import json
import numpy as np
import statsmodels as sm
import math
from osgeo import gdal, ogr, gdalconst
import numpy as np
import os
import sys

import random
import inspect
import pandas as pd

import matplotlib.pyplot as plt

rdriver = gdal.GetDriverByName('GTiff')
vdriver = ogr.GetDriverByName('ESRI Shapefile')



def shapetoRasterSA(inPath,inputName,rType,colName):

    global outputPathPrefix; global nodata; global cell_size; global extent; global srs; global n_rows; global n_cols
    
    print("converting shapefile to raster")
    shpIn = vdriver.Open(inPath,0)
    layer = shpIn.GetLayer()   
    
    ## raster data type
    if colName != None:
        print("Setting raster type from shapefile attribute")
        datColumn = feature.GetField(colName)
        rType = gdal.GetDataTypeByName(str(datColumn.DataType))        

    elif rType != None:
        gdal.GetDataTypeByName(rtype)
        
    else:
        rType = gdal.GDT_Byte
       
    pathOut = outputPathPrefix + "/" + inputName + "_" + str(cell_size) + "_m" + '.tif'
    rasterizedDS = rdriver.Create(pathOut, n_rows, n_cols, 1, rType)
    rasterizedDS.SetGeoTransform([extent[0], cell_size, 0, extent[3], 0, -1 * cell_size])    
    rasterizedDS.SetProjection(srs.ExportToWkt())
    rasterizedDS.GetRasterBand(1).SetNoDataValue(nodata)  
    gdal.RasterizeLayer(rasterizedDS, [1], layer)
    
    band1 = rasterizedDS.GetRasterBand(1)
    list1 = band1.ReadAsArray()
    return pathOut, list1
    
    layer = None
    shpIn = None
    
def proxRaster(inPath,inputName):

    global outputPathPrefix; global nodata; global cell_size; global extent; global srs; global n_rows; global n_cols
    
    # print("converting shapefile to raster")
    # shpIn = vdriver.Open(inPath,0)
    # layer = shpIn.GetLayer()     
    # pathOut = outputPathPrefix + "/intermed_" + inputName + "_" + str(cell_size) + "_m" + '.tif'
    # rasterizedDS = rdriver.Create(pathOut, n_rows, n_cols, 1, gdal.GDT_Byte)
    # rasterizedDS.SetGeoTransform([extent[0], cell_size, 0, extent[3], 0, -1 * cell_size])    
    # rasterizedDS.SetProjection(srs.ExportToWkt())
    # rasterizedDS.GetRasterBand(1).SetNoDataValue(nodata)  
    # gdal.RasterizeLayer(rasterizedDS, [1], layer)
    # rasterizedDS = None
    # layer = None
    # shpIn = None
    
    print("converting raster to distance") # proximity raster   https://gis.stackexchange.com/questions/220753/how-do-i-create-blank-geotiff-with-same-spatial-properties-as-existing-geotiff
    rasIn = gdal.Open(inPath, gdalconst.GA_ReadOnly)
    dist_path = outputPathPrefix + "/" + inputName + "_dist_" + str(cell_size) + "_m" + '.tif'
    proximityDs = rdriver.Create(dist_path, n_rows, n_cols, 1, gdal.GetDataTypeByName("Int32"))   
    proximityDs.SetGeoTransform([extent[0], cell_size, 0, extent[3], 0, -1 * cell_size])    
    proximityDs.SetProjection(srs.ExportToWkt())
    proximityDs.GetRasterBand(1).SetNoDataValue(nodata)
    proximityDs.GetRasterBand(1).Fill(nodata)
    gdal.ComputeProximity(rasIn.GetRasterBand(1), proximityDs.GetRasterBand(1), callback = None)

    band1 = proximityDs.GetRasterBand(1)
    list1 = band1.ReadAsArray()
    list2 = proximityCalc(list1) 
    
    return list2  
    
    pathOut = None
    rasIn = None
    inputName = None
    list1 = None

def rasterSA(inPath,inputName):

    global outputPathPrefix; global nodata; global cell_size; global extent; global srs; global n_rows; global n_cols

    print("resampling raster to study area")
    rasIn = gdal.Open(inPath, gdalconst.GA_ReadOnly)
    inputProj = rasIn.GetProjection()
    inputType = rasIn.GetRasterBand(1)
    inputType = inputType.ReadAsArray().dtype
    
    pathOut = outputPathPrefix + "/" + inputName + "_" + str(cell_size) + "_m" + '.tif'
    #print(pathOut)
    rasterizedDS = rdriver.Create(pathOut, n_rows, n_cols, 1, gdal.GetDataTypeByName(str(inputType)))
    rasterizedDS.SetGeoTransform([extent[0], cell_size, 0, extent[3], 0, -1 * cell_size])    
    rasterizedDS.SetProjection(srs.ExportToWkt())
    rasterizedDS.GetRasterBand(1).SetNoDataValue(nodata)
    gdal.ReprojectImage(rasIn,rasterizedDS,inputProj,srs.ExportToWkt(),gdalconst.GRA_NearestNeighbour)

    band1 = rasterizedDS.GetRasterBand(1)
    list1 = band1.ReadAsArray()
    
    return list1
    
    layer = None
    pathOut = None
    inputName = None
    list1 = None

def normScaler (inArray):
    outArray = (inArray - np.min(inArray)) / (np.max(inArray) - np.min(inArray))
    return outArray
    
def proximityCalc (inArray):
    outArray = np.max(inArray) - inArray
    return outArray

def inputSpatial():

    ## DEBUG
    global responseVar
    responseVar = "Occ"
    ##
    
    # ##global responseVar
    # print('')
    # print("You can enter a series of spatial data files, including a study area shapefile, to be standardized, transformed and stacked to conduct a power analysis for data design. The files must be entered using a json file with the following format.\n")    

    # print("All analysis will be done using a raster stack. The input json file has a field for indicating the file path to a study area shapefile. The projection and extent of the rasters will match that of the input study area shapefile and the cell size is input by the user in the json file. The json also requires the path to an (existing) output folder to which all subseqeuent spatial analytical files will be written.")
    # print('')
    # print("The predictors list has entries for layers to be used in modeling " + responseVar + ", each of which requires a user defined variable name, the path to the input file, the type (vector or raster) and type of analysis transformation, includeing:\n (Scale) values scaled between 0 and 1 \n (Proximity) binary raster or shapefile converted to 'distance-to' \n Etc. \n")
    
    # openJson = input(" (1) Open example json file")
    # if openJson == "1":
        # os.startfile(r'C:\Users\erneilso\OneDrive - NRCan RNCan\Collaborations\ROF\SpatialPowerAnalysis\inJson\inputFile1.json')
    # else:
        # print("Use only numbers available.")
        # readQuestion(strIndex)
        
    print('')
    # print('')
    # inputFile = input(" Paste in path to file of input json: ")
    # if inputFile == "1":
        # inputFile = r'C:\Users\erneilso\OneDrive - NRCan RNCan\Collaborations\ROF\SpatialPowerAnalysis\inJson\inputFile1.json'
    inputFile = r'C:\Users\erneilso\OneDrive - NRCan RNCan\Collaborations\RC Steering Committee\MonitoringDecisionTool\Inputs\SpatIn.json'
    
    # processing json
    f = open(inputFile)
    inData = json.load(f) ;f.close()
    global outputPathPrefix
    outputPathPrefix = inData["outputPathPrefix"]

    ## raster info ##
    global nodata;  global extent; global srs; global n_rows; global n_cols; global cell_size;
    nodata = -9999
    cell_size = inData["cellSize"]
    SAshp = vdriver.Open(inData["SA"],0)  ## this made here: C:/Users/erneilso/OneDrive - NRCan RNCan/Collaborations/ROF/ROF_Map.qgz
    rlayer = SAshp.GetLayer()
    extent = rlayer.GetExtent()
    srs = rlayer.GetSpatialRef()
    n_rows = int(math.ceil(abs(extent[1] - extent[0]) / cell_size))
    n_cols = int(math.ceil(abs(extent[3] - extent[2]) / cell_size)) 
    
    print("The study area has " + str(extent)  + ". \nIt has " + str(n_rows) + " rows and "  + str(n_cols) + " columns." )
    print('')
    
    ## raster info ##
    layers = inData['layers']
    
    global usePredictors
    usePredictors = {}
    global detectionPredictors    
    detectionPredictors = {}
    
    for layer in layers:
    
        ## vector
        if layer["Spat"] == "Vector":
            if layer["Type"] == "Proximity":
                pathOut1, tempArray = shapetoRasterSA(layer["Path"],layer["Name"],rType=None, colName = None) 
                tempArray = proxRaster(pathOut1,layer["Name"])               
                
            elif layer["Type"] == "Binary":
                pathOut1, tempArray = shapetoRasterSA(layer["Path"],layer["Name"],rType=None, colName = None)              
        ## raster        
        else:
            tempArray = rasterSA(layer["Path"],layer["Name"])    
        
        # correct for hypothesized strengthen
        tempArray = normScaler(tempArray)
        tempArray = tempArray * layer["Beta"]

        # add to dictionaries
        if layer["Var"] == "Use":
            usePredictors[layer["Name"]] = tempArray
        else:           
            detectionPredictors[layer["Name"]] = tempArray
    
        print("")
            
            
    ###########################
    ##  PLOTTING
    print("Finished inputing Use vars:", list(usePredictors.keys()))
    for i in usePredictors:
        plt.close()
        pltDat = usePredictors[i]
        plt.title(i + str(np.amin(pltDat)) + "_" + str(np.amax(pltDat)))
        plt.imshow(pltDat)
        plt.show()  

    print("Finished inputing Det vars:", list(detectionPredictors.keys()))
    for i in detectionPredictors:
        plt.close()
        pltDat = detectionPredictors[i]
        plt.title(i + str(np.amin(pltDat)) + "_" + str(np.amax(pltDat)))
        plt.imshow(pltDat)
        plt.show()          
    
def simulateReponse():
    
    global usePredictors
    global detectionPredictors
    
    global outputPathPrefix
    global nodata; global cell_size; global extent; global srs; global n_rows; global n_cols    
    
    global responseValues
    responseValues = {}     
   
    ###########################
    ## PROBABILITY OF USE AS PRODUCT OF INPUT LAYERS (for now)    
    varArray = []
    for i in usePredictors:
        varArray.append(usePredictors[i])      
    varArray = np.array(varArray)    
    responseValues["Use"] = normScaler(np.sum(varArray, axis=0))
    print("Used the inputted rasters to simulate the spatial probability of use across study area")
    
    plt.close()
    pltDat = responseValues["Use"]
    plt.title("Probability of Use " + str(np.amin(responseValues["Use"])) + "_" + str(np.amax(responseValues["Use"])))
    plt.imshow(pltDat)
    plt.show()    

    ###########################
    ## CONVERT TO OCCUPANCY
    ## I want the true occupancy to translate into a proportion of the landscape occupied so the prob of use, which is a function
    ## of covariates, needs to be discretized use trueOcc. So, if trueOcc is 0.2, the top twenty percent of prob use values are occupied.
    global trueOcc
    print('')
    trueOcc = input("Converting use into occupancy. What is true the true proportion of the area that is occupied (number between 0 and 1)?") 
    try:
        trueOcc = float(trueOcc)
        #print("The true occupancy is " + str(trueOcc))
        occThreshold = np.quantile(responseValues["Use"], 1 - trueOcc)
    except:
        print("Enter a number.")
        simulateReponse()

    responseValues["Occupancy"] = np.zeros(shape=responseValues["Use"].shape)
    responseValues["Occupancy"][responseValues["Use"] > occThreshold] = 1     
    pxN = sum( [1 for line in responseValues["Occupancy"] for x in line if x ==1 ] )
    cellArea = cell_size ** 2
    saAreaKM = (cellArea/1000000) * pxN
    print('')
    print("The threshold for predicting occupancy given the probability of use " + str(round(occThreshold,2)) + ". There are " + str(pxN) + " occupied pixels (" + str(saAreaKM) + " km occupied area). This leads to an instantaneous probability of detection in any cell for one, randomly moving, individual of " + str(round(1/pxN,4)))
    
    plt.close()
    pltDat = responseValues["Occupancy"]
    plt.title("Simulated Pixel Occupancy")
    plt.imshow(pltDat)
    plt.show()    
 
    ###########################
    ## SPATIAL PROBABILITY OF USE FOR A POPULATION 
    ##dens = input("What is true density of animals (real number per km2)?") 
    print('')    
    spPop = input("Simulate a population within the occupied cells using a population density. Is the species (1) common , (2) rare, (3) very-rare?")
    if spPop == str(1):
        dens = 0.05
    elif spPop == str(2):
        dens = 0.01
    elif spPop == str(3):
        dens = 0.005
    else:
        print("Please use only the selections displayed.")
        simulateReponse()
        
    global N
    N = float(dens) * saAreaKM
    global popPX
    popPX = N/pxN
    
    print('') 
    print("With a density of " + str(dens) + " individuals per pixel across all occupied pixels, the total population is " + str(round(N,2)) + ". This gives an instantaneous probability of use of any occupied cell, of any randomly moving individual, of " + str(round(popPX,4)))
    
    ###########################
    ## SPATIAL PROBABILITY OF DETECTION    
    ## get the product of occupancy and use to extract the probability of use at occupied sites only
    detArray = []
    for i in responseValues:
        detArray.append(responseValues[i])    
    detArray = np.array(detArray)
    spatDet = normScaler(np.prod(detArray, axis=0))   
    spatDet = spatDet * popPX

    # scaling by actual detection values
    if len(detectionPredictors) != 0:    
        detArray = []
        for i in detectionPredictors:
            detArray.append(normScaler(detectionPredictors[i]))      
        detArray = np.array(detArray)
        spatDet1 = normScaler(np.sum(detArray, axis=0)) 
        spatDet1 = spatDet1 * spatDet    
    else:
        spatDet1 = spatDet

    global meanDetection
    meanDetection = np.mean(spatDet1)
    responseValues["Detection"] = spatDet1  

    print('')   
    print("Given the spatial variation in probability of use for the population and variation in detection at each site, the mean instantaneous probability of detection across occupied cells, for any randomly moveing individual, is " + str(round(meanDetection,4)))    
    
    plt.close()
    pltDat = responseValues["Detection"]
    plt.title("Simulated Pixel Detectability " + str(np.amin(responseValues["Detection"])) + "_" + str(np.amax(responseValues["Detection"])))
    plt.imshow(pltDat)
    plt.show()  
    
    # ###########################
    # ##  PLOTTING
    # #print('')
    # #print("Finished calculating response vars:", list(responseValues.keys()))
    # print('')
    # print("RESPONSE PREDICTORS")
    # for i in responseValues:
        # plt.close()
        # pltDat = responseValues[i]
        # plt.title(i + str(np.amin(pltDat)) + "_" + str(np.amax(pltDat)))
        # plt.imshow(pltDat)
        # plt.show()    

    ###########################
    ## USE RASTER
    responsePath = outputPathPrefix + "/" + r'SimulatedUse.tif'
    rasterizedDS = rdriver.Create(responsePath, n_rows, n_cols, 1, gdal.GetDataTypeByName("Float32"))
    rasterizedDS.SetGeoTransform([extent[0], cell_size, 0, extent[3], 0, -1 * cell_size])    
    rasterizedDS.SetProjection(srs.ExportToWkt());rasterizedDS.GetRasterBand(1).SetNoDataValue(nodata);rasterizedDS.GetRasterBand(1).Fill(nodata)
    rasterizedDS.GetRasterBand(1).WriteArray(responseValues["Use"])
    
    ## DETECITON RASTER
    responsePath = outputPathPrefix + "/" + r'SimulatedDetection.tif'
    rasterizedDS = rdriver.Create(responsePath, n_rows, n_cols, 1, gdal.GetDataTypeByName("Float32"))
    rasterizedDS.SetGeoTransform([extent[0], cell_size, 0, extent[3], 0, -1 * cell_size])    
    rasterizedDS.SetProjection(srs.ExportToWkt());rasterizedDS.GetRasterBand(1).SetNoDataValue(nodata);rasterizedDS.GetRasterBand(1).Fill(nodata)
    rasterizedDS.GetRasterBand(1).WriteArray(responseValues["Detection"])
    
    ## OCCUPANCY RASTER
    responsePath = outputPathPrefix + "/" + r'SimulatedOccupancy.tif'
    rasterizedDS = rdriver.Create(responsePath, n_rows, n_cols, 1, gdal.GetDataTypeByName("Float32"))
    rasterizedDS.SetGeoTransform([extent[0], cell_size, 0, extent[3], 0, -1 * cell_size])    
    rasterizedDS.SetProjection(srs.ExportToWkt());rasterizedDS.GetRasterBand(1).SetNoDataValue(nodata);rasterizedDS.GetRasterBand(1).Fill(nodata)
    rasterizedDS.GetRasterBand(1).WriteArray(responseValues["Occupancy"])

def pixel(file,dx,dy):
    gt = file.GetGeoTransform()
    px = gt[0]
    py = gt[3]
    rx = gt[1]
    ry = gt[5]
    x = px + rx * dx
    y = py + ry * dy
    return x,y
    
def simulateOccupancyData():
    
    global nodata; global cell_size; global extent; global srs; global n_rows; global n_cols
    global responseValues
    global prevPos
    global responseVar
    
    ## LOOP OVER STUDY DESIGN SCENARIOS
    
    ## sites
    siteScenN = input("Enter the number of site scenarios.")
    try:
        siteScenN = int(siteScenN)
    except:
        print("Enter a number.")
        simulateOccupancyData()  
    maxCam = input("Enter the max number of cameras.")
    try:
        maxCam = int(maxCam)
    except:
        print("Enter a number.")
        simulateOccupancyData() 
    minCam = input("Enter the min number of cameras.")
    try:
        minCam = int(minCam)
    except:
        print("Enter a number.")
        simulateOccupancyData()
    sitesN = range(minCam,maxCam,round(maxCam/siteScenN))    
    #sitesN = range(10,50,round(50/6))
    
    ## durations
    durScenN = input("Enter the number of duration scenarios.")
    try:
        durScenN = int(durScenN)
    except:
        print("Enter a number.")
        simulateOccupancyData()          
    maxDur = input("Enter the max duration of deployments (weeks).")
    try:
        maxDur = int(maxDur)
    except:
        print("Enter a number.")
        simulateOccupancyData()     
    minDur = input("Enter the min duration of deployments (weeks).")
    try:
        minDur = int(minDur)
    except:
        print("Enter a number.")
        simulateOccupancyData()      
    dursN = range(minDur,maxDur,round(maxDur/durScenN))    
    #dursN = range(5,20,round(20/4))

    
    # camConfig = input("Enter the configuration of cameras \n (1) Systematic\n (2) Random\n (3) Stratigied Random \n")    
    # if camConfig.upper() == "B":
        # readQuestion(prevPos)    
    # if camConfig != "1":
        # print("Not available.")
        # simulateOccupancyData()
    # print('')
    
    # camN = 20
    # camConfig = "Systematic" 
        


    ## make output folder for deteciton histories
    import shutil
    if os.path.isdir(outputPathPrefix + '/DetectionHistories'):
        shutil.rmtree(outputPathPrefix + '/DetectionHistories')
    os.makedirs(outputPathPrefix + '/DetectionHistories') 
    
    for sn in sitesN:
    
        ###########################
        ## SYSTEMATIC SITES
        rArray = responseValues["Use"]
        rSize = rArray.size
        rSysN = int(round(rSize / sn))
        ##print("every " + str(rSysN))
        
        #print("Setting pixels as sites.")    
        siteList = np.zeros(shape=responseValues["Use"].shape)
        siteList = np.ndarray.flatten(siteList)
        siteList[1::rSysN] = 1
        siteList = np.reshape(siteList,(responseValues["Use"].shape))
        
        # global siteOcc
        # siteOcc = []
        # occArray = np.ndarray.flatten(responseValues["Occupancy"])
        # siteOcc = occArray[1::rSysN]
        # print("Site occupancy is : \n" + str(siteOcc))  
        
        # global siteDet
        # siteDet = []
        # detArray = np.ndarray.flatten(responseValues["Use"])
        # siteDet = detArray[1::rSysN]
        # #siteDet = siteDet * siteOcc
        # print("Site detectability is : \n" + str(siteDet))  

        # print("Populated site occupancy and detectability.")
        

        ###########################
        ## CREATE SPATIAL OUTPUTS AND LIST OF POINTS
        
        # RASTER
        # rasterName = outputPathPrefix + "/" +  "SampleSitesRaster.tiff"
        # sitesDS = rdriver.Create(rasterName, n_rows, n_cols, 1, gdal.GetDataTypeByName("Int32"))
        # sitesDS.SetGeoTransform([extent[0], cell_size, 0, extent[3], 0, -1 * cell_size])    
        # sitesDS.SetProjection(srs.ExportToWkt())
        # sitesDS.GetRasterBand(1).SetNoDataValue(nodata)
        # sitesDS.GetRasterBand(1).Fill(nodata)
        # sitesDS.GetRasterBand(1).WriteArray(siteList) 

        # # SHAPEFILE - https://gis.stackexchange.com/questions/268395/converting-raster-tif-to-point-shapefile-using-python
        # print("exporting shapefile of cameras")
        # shpName = outputPathPrefix + "/" +  "SampleSitesPoints.shp"
        # outDataSource = vdriver.CreateDataSource(shpName)
        # outLayer = outDataSource.CreateLayer(shpName, srs, geom_type=ogr.wkbPoint )
        # featureDefn = outLayer.GetLayerDefn() 
           
        # # attributes
        # outLayer.CreateField(ogr.FieldDefn("Ind", ogr.OFTInteger))
        # outLayer.CreateField(ogr.FieldDefn("Occ", ogr.OFTInteger))
        # outLayer.CreateField(ogr.FieldDefn("Det", ogr.OFTReal))
        
        ## GET DETECTION AND OCCUPANCY AT SITES
        pointInds = []
        siteOcc = []
        siteDet = []        
        #point = ogr.Geometry(ogr.wkbPoint) 
        siteIndex = 0
        for rInd, row in enumerate(siteList):
            for cInd, value in enumerate(row):
            
                if value == 1:
                    # print(siteList[rInd,cInd])
                    # print(value)
                    
                    siteIndex += 1
                    # print(siteIndex)
                    pointInds.append(siteIndex)
                    OccValue = responseValues["Occupancy"][rInd,cInd]
                    DetValue = responseValues["Detection"][rInd,cInd]
                    siteOcc.append(OccValue)
                    siteDet.append(DetValue)
                    # print(str(OccValue))
                    # print(str(DetValue))
                    
                    # get covatriates here 
                    
                    # # shapefile
                    # Xcoord, Ycoord = pixel(sitesDS,cInd,rInd)
                    # # print(Xcoord);print(Ycoord)
                    # point.AddPoint(Xcoord, Ycoord)
                    # outFeature = ogr.Feature(featureDefn)
                    # outFeature.SetGeometry(point)
                    # outFeature.SetField("Ind", int(siteIndex))
                    # outFeature.SetField("Occ", int(OccValue))
                    # outFeature.SetField("Det", float(DetValue))
                    # outLayer.CreateFeature(outFeature)
                    # outFeature.Destroy()
        
        siteOcc = np.array(siteOcc)
        siteDet = np.array(siteDet)
        # print("True occupancy at cameras is:")
        # print(siteOcc)
        # print('')
        
        # print("True probabilities of detection at cameras are:")
        # print(siteDet)
        # print('')        
        
        ###########################   
        ## GET NUMBER OF SITES VISITS (NO MISSING DATA FUNCTIONS YET)

        for dn in dursN:             
            ScenName = str(sn) + "_" + str(dn)
            os.makedirs(outputPathPrefix + '/DetectionHistories/' + ScenName)    

            simN = 10        
            for simn in range(simN):        
                DetectionHistory = []
                
                for sd in siteDet:    
                    siteDetHist = []
                    
                    for sv in range(dn):        
                        det = 0
                        samp = random.uniform(0, 1)
                        if samp < sd:
                            det = 1            
                        siteDetHist.append(det)    
                        
                    DetectionHistory.append(siteDetHist)

                ## export 
                dh = pd.DataFrame(DetectionHistory)
                dh.to_csv(outputPathPrefix + '/DetectionHistories/' + ScenName + "/" + str(simn + 1 ) + '_dh.csv', index=False)  

def readQuestion(dat, strIndex):
    
    print('')
    global currPos
    global prevPos
    global answer    
    
    currPos = dat[strIndex]    
    
    answer = input( currPos["Text"] )        
        
    if answer.upper() == "B":
        readQuestion(dat, prevPos)
    
    if currPos["Function"] != "":        
        #print("Calling " + currPos["Function"] ) 
        method = globals() [ currPos["Function"] ]
        # mArgs = inspect.getfullargspec(method)
        # print(mArgs)        
        method()  
        
    if answer in currPos["posAnswers"]:
    
        if currPos["ResponseIndices"] == ["-999"]:
            print("exit tool")
        else:        
            prevPos = strIndex
            nextPos = currPos["posAnswers"].index(answer)
            readQuestion(dat, currPos["ResponseIndices"][nextPos] )
            
    else:
        print("Please use only the selections displayed.")
        readQuestion(dat, strIndex)
 
def addResponseVar():

    global answer
    global responseVar
    
    if answer == "1":
        responseVar = "Use"
    elif answer == "2":
        responseVar = "Range"
    else:
        responseVar = "Occupancy"
        print("modeling occupancy")

            








