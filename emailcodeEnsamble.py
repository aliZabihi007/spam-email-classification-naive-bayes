import csv
import numpy as np
import random
from keras  import layers , models,Sequential 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
import math 

#تابع تعریف مدل شبکه عصبی می باشد که ورودی تابع همون کوئری ورودی به شبکه خواهد بود 
def mlp(testqu):

  #در این قسمت به باز کردن فایل و برسی ایمیل های داخل فایل می پردازش

    with open('emails.csv', newline='') as csvfile:
        Datatotal = list(csv.reader(csvfile))

   
#ارایه هایی تعریف می شود تحت عنوان ارابه آموزشی و آرایه تست
   
    matrixtrain = []
    matrixtest = []
    matrixtrainy = []
    matrixtesty = []

#به صورت رندوم داد های آموزشی 3500 داده آموزشی برداشته و در داخل ارایه آموزشی قرار میگیرد
    number = random.sample(range(1,5000), 3500)
  
    for i in number:
        matrixtrain.append(Datatotal[i][1:3001])
        matrixtrainy.append(Datatotal[i][3001])

#داد های آموززشی به صورت رندوم برای داده های تست انتخاب می شود  و هم داده های تست به همراه مقدار  خروجی هر نوع داده
    number = random.sample(range(1,5000), 1000)
    for i in number:
        matrixtest.append(Datatotal[i][1:3001])
        matrixtesty.append(Datatotal[i][3001])
  
#برسی داده های آموزشی انتخاب شده از نظر ابعاد 
    matrixtrain=np.array(matrixtrain)
    matrixtest=np.array(matrixtest)
    matrixtrainy=np.array(matrixtrainy)
    matrixtesty=np.array(matrixtesty)

    print(matrixtrain.shape)
    print(matrixtest.shape)

    print(matrixtrainy.shape)
    print(matrixtesty.shape)

#در ااین بخش نرمال سازی صورت میگیرد و نرمال سازی بدین صورت میباشد که بخش هایی که کلمه وجود دارد و تکرار شده را با 1 و جاهایی که استفاده نشده با 0 نمایش در می آید 
    for i in range(0, 3500, 1):
      for j in range(0, 3000, 1):
        if(int(matrixtrain[i][j])>0):
          matrixtrain[i][j]=1
      else:
          matrixtrain[i][j]=0

#روند بالا  برای داده های تست در پایین انجام می شود که داده های تست مقدار صفر و مقدار یک نسبت به هم شناسایی و در ماتریس و ارایه قرار خواهد گرفت
    for i in range(0, 1000, 1):
      for j in range(0, 3000, 1):
        if(int(matrixtest[i][j])>0):
          matrixtest[i][j]=1
      else:
          matrixtest[i][j]=0  

#نرمال سازی و پیش پردازش مورد نظر برای ورود به شبکه 
    matrixtrain = matrixtrain.astype(np.float32)
    matrixtrainy = matrixtrainy.astype(np.float32)

    

    matrixtest = matrixtest.astype(np.float32)
    matrixtesty = matrixtesty.astype(np.float32)

    matrixtrain=matrixtrain/10.0
    matrixtest=matrixtest/10.0

    Y_train = to_categorical(matrixtrainy, 2)
    Y_test = to_categorical(matrixtesty, 2)

#مدل در این قسمت تعریف می شود که خروجی دو کلاسه که نشان از اسپم بودن و نبودن را میدهد ورودی 3000 کلمه ای را دارد 
    model = Sequential()
    model.add(Dense(20, input_shape=(3000,), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))

#چون یک جور کار ما عمل کلاس بندی می باشد از روش  از تابع بهینه ساز استفاده خواهیم کرد و توابع بهینه ساز زیادی مورد استفاده قرار گرفته و نتیجه بهتری در این روش بدت آمده است و در 20 ایپاک اجرای می شود 
#و همواره مقداری را جداسازی می کنیم تحت عنوان برسی اورفیت شدن شبکه 
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(matrixtrain,Y_train,epochs=20,validation_split=0.2)
    model.evaluate(matrixtest,Y_test)

 #ورودی تابع را در این بخش شروع به عمل پیش بینی میکنیم یعنی پنجاه داده ورودی را به شبکه میدهیم و خروجی را دوباره   به تابع اصلی برگشت می دهیم    
    print(testqu.shape)
    predicted=model.predict(testqu)
    print("ssssssssssssssss")
    print(predicted)
    ret=list()

    #مقدار بیشینه را بدست آورده و به عنوان خروجی در ارایه ذخیره  و برگشت می دهیم 
    for i in range(0,50,1):
       ret.append(np.argmax(predicted[i]))
    print(matrixtesty[20])
    return ret

#تابع برای اجرای شبکه بر اساس نزدیک ترین همسایه می باشد که ورودی همان کوئری ورودی به شبکه یالا تر بوده 
def Knn(testqu):
  
  #در ابتدا دوباره مراحل خواندن فایل و تجزیه و ذخیره در آرایه ها را داریم 
  with open('emails.csv', newline='') as csvfile:
          Datatotal = list(csv.reader(csvfile))


#ارایه ها آموزش و تست
  matrixtrain = []
  matrixtest = []
  matrixtrainy = []
  matrixtesty = []

#مقادیر رندوم برای داده های تست و آموزشی
  number = random.sample(range(1,5000), 4500)
  print(number)
  for i in number:
          matrixtrain.append(Datatotal[i][1:3001])
          matrixtrainy.append(Datatotal[i][3001])

#برای داده های  تست این رندوم گیری صورت میگیرد 
  number = random.sample(range(1,5000), 100)
  for i in number:
          matrixtest.append(Datatotal[i][1:3001])
          matrixtesty.append(Datatotal[i][3001])  print(number)

#نرمال سازی و برسی خروجی هایی داده ای پردازش شده در مرحله های قبلی 
  matrixtrain=np.array(matrixtrain)
  matrixtest=np.array(matrixtest)
  matrixtrainy=np.array(matrixtrainy)
  matrixtesty=np.array(matrixtesty)

  print(matrixtrain.shape)
  print(matrixtest.shape)

  print(matrixtrainy.shape)
  print(matrixtesty.shape)

#در این مرحله سعی شده  کلماتی که در ایمیل یا پیام خاصی استفاده شده را به صورت عدد یک و عدد صفر یک جور کلاس بندی صورت دهد 
#بخش نرمال سازی برای مرحله بعدی را در این مرحله انجام می شود
  for i in range(0, 4500, 1):
        for j in range(0, 3000, 1):
          if(int(matrixtrain[i][j])>0):
            matrixtrain[i][j]=1
          else:
            matrixtrain[i][j]=0



#ارایه هایی تعریف می شود 
  matrixtrain = matrixtrain.astype(np.float32)
  matrixtrainy = matrixtrainy.astype(np.float32)


  ret=list()

#در این بخش  سعی دارد با عمل فاصله اقیدوسی گرفت و سعی دارد یک کوئری از داخل لیست خارج میکند و در فضای داده های آموزشی قرار می دهد و فاصله خود را نسبت به همه بدست می آورد 
#پس سعی دارد  در مسیری   فاصله خود را با همه داده های آموزشی  حساب می کنیم و آن 5 داده ای که نزدیک تر از به داده تستی آن را   انتخاب می کند  و آن ها را به صورت نزولی مرتب میکند 
#و بعد آنها را که کمترین است را جدا می کند 
  for t in range(0,50,1):
    cnter=testqu[t]
    rsafe,wsafe=matrixtrain.shape
    distance=0
    disarray=[]
    for i in range(0, rsafe, 1):
      distance=0
      for j in range(0, wsafe, 1):
        distance=distance + math.sqrt(((matrixtrain[i][j]-cnter[j])**2))
      disarray.append(distance)
      
   
    disarray= np.array(disarray) 
    indices = disarray.argsort() 
    indices=indices[0:5]
    plase=0
    negetive=0

# در این بخش برسی میکنیم بر اساس برچسب ها و تعداد نقاط منفی و تعداد پرچسب های مثبت برسی میکنیم کدام بیشترین تعداد فاصله به کوئری دارد 
    for i in indices:
      out=matrixtrainy[i]
      if(out==0):
          negetive+=1
      else:
          plase+=1

    if(negetive>plase):
        ret.append(0)
    else:
        ret.append(1)
    print("///////////////////////////") 
  return ret

#/e///////////////////main//////////////////////////////////////////


#در این بخش تست می باشد کوئری ها مون را می سازیم و به عنوان ورودی به شبکه می دهیم 
with open('emails.csv', newline='') as csvfile:
          Datatotal = list(csv.reader(csvfile))

test = []
testy = []

#اعدادی رندوم 50 عدد
number = random.sample(range(1,5000), 50)
for i in number:
          test.append(Datatotal[i][1:3001])
          testy.append(Datatotal[i][3001])

#ذخیره در داخل ارایه تست هم برچسب هم داده 
test=np.array(test) 
testy=np.array(testy)

#انجام پیش پردازش برای  اجرای شدن
for i in range(0, 50, 1):
      for j in range(0, 3000, 1):
        if(int(test[i][j])>0):
          test[i][j]=1
        else:
          test[i][j]=0

test = test.astype(np.float32)
testy = testy.astype(np.float32)
test=test/10.0


#ورود داده ها به مدل  شبکه عصبی آموزش و برداشت خروجی
resu1=mlp(test)
test=test*10.0
print(resu1)

#اجرای مدل دیگر و ارسال داده و کوئری
resu2=Knn(test)
print(resu2)

#دو خروجی از مدل ها مقدار ماکزیمم هر کدام را انتخاب میکند به صورت درایه به درایه این کار صورت میگیرد 
finallyarray=list()
finallyarray=np.maximum(resu1,resu2)

print(finallyarray)
print(testy)


accurat=0
error=0

#با برابر بودن مقدار خروجی ماکزیمم و مقدار خروجی واقعی میزان خطا و میزان خطا بدست می آید
for i in range(0,50,1):

  if(finallyarray[i]==testy[i]):
    accurat+=1
  else:
    error+=1

print(accurat)
print(error)

