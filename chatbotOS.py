import urllib3
import json
import pandas as pd
from banggatest import *


def adminArea():
    data_df=pd.read_csv("./location.csv",header=0)
    guSet,dongSet=set(data_df['자치구명']),set(data_df['법정동명'])
    gudong=data_df[['자치구명','법정동명']]
    guDongDict={k:g["법정동명"].tolist() for k,g in gudong.groupby("자치구명")}
    
    return guDongDict,guSet,dongSet



def chatbot(text):
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
    accessKey= "b0c887cb-30e0-432b-abfd-9b3055485fb4"
    analysisCode = "ner"
    qDict={}
    adminDict,guSet,dongSet=adminArea()
    print("질문:",text)

    requestJson = {
        "access_key": accessKey,
        "argument": {
            "text": text,
            "analysis_code": analysisCode
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )

    print("[responseCode] " + str(response.status))
    #print("[responBody]")
    print(str(response.data,"utf-8"))
    result=str(response.data,"utf-8")
    a=json.loads(result)
    
    ne=a["return_object"]["sentence"][0]["NE"]
    #word=a["return_object"]["sentence"][0]["word"]
    WSD=a["return_object"]["sentence"][0]["WSD"]

    address=[]
    quantity=[]

    for i in ne:
        if i["type"] in ("LCP_COUNTY","LCP_CITY"):
            print(i["text"])
            address.append(i["text"])
        
        if i["type"] in ("QT_OTHERS","QT_SIZE","QT_LENGTH","QT_COUNT","QT_PRICE"):
            qDict["평수"]=i["text"]

    for i in WSD:
        if i["text"] in ("전세","월세"):
            qDict["전월세"]=i["text"]
        if i["text"] in ("아파트","오피스텔","빌라"):
            qDict["거주 유형"]=i["text"]


    for i in address:
        print(i)
        if i in guSet:
            qDict["구"]=i
        if i in dongSet:
            qDict["동"]=i
    
    if "동" in qDict.keys() and "구" not in qDict.keys():
        for gu, dong in adminDict.items():
            for i in dong:
                if i == qDict["동"]:
                    qDict["구"]=gu
    
      
    
    return qDict


    

    """
    입력:강남구 반포동 아파트 30평 전세 시세 알려주세요
    반환:qDict
    {'address': ['강남구', '반포동'],
     'payment': '전세',
     'quantity': ['30평'],
     'type': '아파트'}
    """




def chk_dict(dic):
    keys=["구","동","전월세","평수","거주 유형"]
    count=len(keys)
    while True:
        for i in keys:
            if i not in dic.keys():
                dic[i]=input("{}를 입력해주세요.".format(i))
            count-=1     
        if len(dic.keys())==len(keys):
            return dic


               
"""
@pyqtSlot()
    def analysis(self):
        d=ch_dic(chatbot())
        if d["전월세"]=="전세":
            result=jeonse(d["구"]+d['동'],d['평수'][:-1],d['주거형태'])
        else:
            result=wolse(d["구"]+d['동'],d['평수'][:-1],d['주거형태'])
        self.answer.setText(str(result))

"""






