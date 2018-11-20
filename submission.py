import helper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def new_tokenizer(s):
    return s.split()
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy() 
    parameters={}
    parameters.update({'C':2,'kernel':'linear','coef0':0,'degree':0,'gamma':'auto'})
    X=[]
    for e0 in strategy_instance.class0:
        s=" ".join(e0)
        X.append(s)
    for e1 in strategy_instance.class1:
        s=" ".join(e1)
        X.append(s)
    #print(X)
    vectorizer=CountVectorizer(tokenizer=new_tokenizer)
    train=vectorizer.fit_transform(X)
    transformer=TfidfTransformer(use_idf=False)
    Xtrain=transformer.fit_transform(train)
    Y=[]
    for i in range(0,len(strategy_instance.class0)):
        Y.append(0)
    for i in range(0,len(strategy_instance.class1)):
        Y.append(1)
    clf=strategy_instance.train_svm(parameters,Xtrain,Y)
    #print(len(vectorizer.get_feature_names()))
    W=clf.coef_.toarray()[0]
    #print(W)
    dic_p={}
    dic_n={}
    for i in range(0,len(W)):
        dic_p.update({vectorizer.get_feature_names()[i]:W[i]})

    
 
    L_delete=sorted(dic_p.items(),key=lambda item:item[1], reverse=True)
    # L_add=sorted(dic_p.items(),key=lambda item:item[1], reverse=False)
    delete_words=[]
    # add_words=[]
    for e in L_delete:
        delete_words.append(e[0])
##    for e in L_add:
##        add_words.append(e[0])
    #print(delete_words[])
        
    
    
    #print(L_delete)
    with open(test_data,'r') as file:
        orignal_data=[line.strip().split(' ') for line in file]
    
    times=[0]*len(orignal_data)
    #print('o',orignal_data[0])
    words_to_delete=[]
    for i in range(len(orignal_data)):
        words_to_delete.append([])
##    words_to_add=[]
##    for i in range(len(orignal_data)):
##        words_to_add.append([])
##    
    #print(words_to_delete)
    
    #print(len(orignal_data))
    
    for word in delete_words:
        #print(word)
        for i in range(len(orignal_data)):
            if times[i]!=20:
                if word in orignal_data[i]:
                    words_to_delete[i].append(word)
                    times[i]+=1
##    times1=[0]*len(orignal_data)
##    for word in add_words:
##        for i in range(len(orignal_data)):
##            if times1[i]!=5:
##                if word not in orignal_data[i]:
##                    words_to_add[i].append(word)
##                    times1[i]+=1
    #print(len(words_to_delete))

    #print(times)
    #print(orignal_data)
    #print(words_to_delete[0][1])
    
    
    #print(times)
    save_data=[]
    for i in range(len(orignal_data)):
        save_data.append([])

    for i in range(len(orignal_data)):
        for w in orignal_data[i]:
            if w not in words_to_delete[i]:
                save_data[i].append(w)
    
##    for i in range(len(orignal_data)):
##        for w in words_to_add[i]:
##            save_data[i].append(w)
                    
            
            
##    for i in range(len(orignal_data)):
##        print(len(orignal_data[i])-len(save_data[i]))

    #print('rm',orignal_data[0])
    
    modi=[]
    for i in range(len(save_data)):
        modi.append(" ".join(save_data[i])+" "+'\n')
    
    #print(modi[0])
   
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...

        
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    with open(modified_data,'w') as outfile:
        for i in range(len(modi)):
            outfile.write(modi[i])

    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.

fool_classifier('./test_data.txt')
