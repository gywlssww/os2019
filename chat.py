import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from chatbotOS import *

class Chat(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = loadUi('./chat.ui',self)
        self.setupUi()
        self.setupFunc()
    
    def setupUi(self):
        self.setWindowTitle("BanggaBangga")
        
    def setupFunc(self):
        self.sendBtn.clicked.connect(self.showQuestion)
        self.sendBtn.clicked.connect(self.analysis)
        #self.question.clicked.connect(self.showGraph)
        
    def showQuestion(self):
        q=self.inputTxt.toPlainText()
        self.question.setText(q)
    
    def showAnswer(self,money):
        self.answer.setText(str(money)+"입니다. 클릭하면 도표를 보실 수 있습니다.")
        self.inputTxt.clear()

    #def showGraph(self):
        
    def analysis(self):
        text=self.inputTxt.toPlainText()
        print(text)
        d=chk_dict(chatbot(text))
        if d["전월세"]=="전세":
            result=jeonse(d["구"]+d['동'],d['평수'][:-1],d['거주 유형'])
            print(str(result))
            print("결과",result)
            self.showAnswer(result)
        else:
            result=wolse(d["구"]+d['동'],d['평수'][:-1],d['거주 유형'])
            self.showAnswer(result)

if __name__ =="__main__":
    app = QApplication(sys.argv)
    mywindow = Chat()
    mywindow.show()
    app.exec_()




