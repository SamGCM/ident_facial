
import face_recognition
import numpy as np
import cv2
import os

from datetime import datetime

class Recog_face(object):
    def __init__(self):
        super.__init__()
        path = 'images'
        images = []
        classNames = []
        mylist = os.listdir(path)

        #ADICIONA IMAGENS NA LISTA 'IMAGES'

        for cl in mylist:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        print(classNames)

        #CODIFICA AS IMAGENS DENTRO DA LISTA 'IMAGES'

        def findEncoding():
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList

        #EM ARQUIVO CSV ADICIONA O NOME, DATA E HORÁRIO DE ENTRADA OU RECONHECIMENTO FACIAL

        def markEntrada(name):
            with open('Controle.csv', 'r+') as f:
                myDataList = f.readlines()
                namelist = []
                for line in myDataList:
                    entry = line.split(',')
                    namelist.append(entry[0])
                if name not in namelist:
                    now = datetime.now()
                    dtString = now.strftime('%d/%m/%Y')
                    hsString = now.strftime('%H:%M')
                    f.writelines(f'\n{name};{dtString};{hsString}')





        encodeListKnow = findEncoding()
        print('Codificação completa')

        #ACESSO A WEBCAM
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) #AJUSTE DE RESOLUÇÃO
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #AJUSTE DE COR

            facesCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace,faceLoc in zip(encodeCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)


                #RETANGULO QUE SURGE NO ROSTO IDENTIFICADO

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))

                    markEntrada(name)

                    #Acrescentado recentemente para fechar a janela com comando
                    if name in classNames:
                        if cv2.waitKey(1) & 0xff == ord('q'):

                            cap.release()
                            cv2.destroyAllWindows()
                            break




            cv2.imshow('Webcam',img)
            cv2.waitKey(1)
















