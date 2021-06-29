class machineLearningModel:
    def optimize(this,data,label,typ='train'):
        from numpy import nan,array,where
        from pandas import DataFrame
        from statistics import mode
        if(typ == 'train'):
            code=dict()
            data = data.copy()
            __pipe = dict()
            for column in  data.columns:
                if(data[column].dtype == 'O'):
                    code[column] = dict()
                    index_list = array(sorted(data[column].value_counts().index))
                    for value in index_list:
                        if(value not in  [nan]+this.missing):
                            ind = where(index_list == value)[0][0]
                            code[column][value] = ind
                            data[column] = data[column].replace(value,ind)
                for miss in this.missing:
                    data[column] = data[column].replace(miss,nan)
                __pipe[column] = {'mean':data[column].mean(),"std":data[column].std(),"median":data[column].median(),"zero":0,'mode':mode(data[column])}
                data[column] = data[column].replace(nan,__pipe[column][this.replace[column]])
                if(column != this.label):
                    data[column] = (data[column]-__pipe[column]["mean"])/__pipe[column]["std"]
            this.__pipe=__pipe
            dataFeature = data.drop(this.label,axis = 1) 
            dataLabel = data[this.label].copy()
            return dataFeature,dataLabel,code
        elif(typ == 'test'):
            data = data.copy()
            for column in data.columns:
                if(data[column].dtype == 'O'):
                    for value in set(data[column]):
                        if(value not in [nan]+this.missing):
                            ind = this.code[column][value]
                            data[column] = data[column].replace(value,ind)
                for miss in this.missing+[nan]:
                    data[column] = data[column].replace(miss,this.__pipe[column][this.replace[column]])
                if(column != this.label):
                    data[column] = (data[column]-this.__pipe[column]["mean"])/this.__pipe[column]["std"]
            if(label != False):
                data,label = data.drop(this.label,axis = 1) ,data[this.label].copy()
            return data,label
        elif(typ == 'prediction'):
            df = DataFrame([data],columns=list(this.__pipe.keys())[:-1])
            return this.optimize(df,label=False,typ='test')
    
    def optimizePipline(this,feature,label = None,typ='train'):
        from numpy import nan
        if (typ == 'train'):
            dataFeature,dataLabel,code = this.optimize(feature,label)
            return dataFeature,dataLabel,code
        elif(typ == 'test'):
            dataFeature,dataLabel = this.optimize(feature,label,typ='test')
            return dataFeature,dataLabel
        elif(typ == 'prediction'):
            dataFeature,label = this.optimize(feature,False,typ)
            return dataFeature
        
    def __repr__(this):
        return this.desc
    def predict(this,feature,getCode=False):
        def labelRecogniser(code,getCode):
            if this.label in this.code and not getCode:
                return list(this.code[this.label].keys())[code]
            else:
                return code
        if(len(feature)==1):
            return labelRecogniser(this.__model.predict(this.optimizePipline(feature[0],label=None,typ='prediction').to_numpy())[0],getCode=getCode)
        else:
            return [labelRecogniser(this.__model.predict(this.optimizePipline(sampleFeature,label=None,typ='prediction').to_numpy())[0],getCode=getCode) for sampleFeature in feature]
    def performance(this):
        print(this)
        print("f1 score                      ",this.f1Score)
        print("precision score               ",this.precisionScore)
        print("recall score                  ",this.recallScore)
        print("confusion matrix              ",this.confusionMatrix.tolist())
        print("averge time per prediction    ",this.speed,"sec")
    def __init__(this,model,trainFeature,label,testFeature,desc=None,replace="median",missing=['?']):
        this.desc = desc
        this.label = label
        this.missing=missing
        if(type(replace)==str):
            replace=[replace]*len(trainFeature.columns)
        this.replace = {trainFeature.columns[column] : replace[column] for column in range(len(trainFeature.columns))}
        def trainingAndTesting(model,trainFeature,label,testFeature):
            from sklearn.metrics import mean_squared_error,precision_score, recall_score,confusion_matrix,f1_score
            from sklearn.model_selection import cross_val_score,cross_val_predict
            from numpy import sqrt
            from time import time
            trainFeature,trainLabel,this.code = this.optimizePipline(trainFeature,label)
            
            this.value = {this.code[this.label][x]:x  for x in this.code[this.label]}
            model.fit(trainFeature,trainLabel)
            this.__model = model
            ##speed calculation
            ini=time()
            this.predict(testFeature.to_numpy()[:1000,:-1])
            this.speed = (time()-ini)/1000
            ###################
            testFeature,testLabel = this.optimizePipline(testFeature,label,typ = 'test')
            predict = this.__model.predict(testFeature)
            error = sqrt(mean_squared_error(testLabel, predict))
            this.RMSE = error
            this.confusionMatrix = confusion_matrix(testLabel,predict)
            this.precisionScore = precision_score(testLabel, predict)
            this.recallScore = recall_score(testLabel,predict)
            this.f1Score = f1_score(testLabel, predict)
        trainingAndTesting(model,trainFeature,label,testFeature)