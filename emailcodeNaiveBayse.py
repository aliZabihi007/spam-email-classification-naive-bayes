import csv
import numpy as np

# در این قسمت فایل دریافتی را به صورت مرحله به مرحله و خط به خط ایمیل ها را میخوانیم و در داخل ارایه قرار می دهیم  که ارایه ای با اندازه 5173
with open('emails.csv', newline='') as csvfile:
    Datatotal = list(csv.reader(csvfile))

# //////////////////////////////////////////
# در کد زیر ارایه هایی برای جدا سازی کردن بر اساس خروجی که دایمیل های اسپم شده و ایمیل های اسپم نشده
matrixspam = []
matrixsafe = []
for i in range(1, 5173, 1):
    out = Datatotal[i][3001]
    if (int(out) == 0):
        matrixspam.append(Datatotal[i])
    else:
        matrixsafe.append(Datatotal[i])
# //////////////////////////////////////////
# در این قسمت آرایه هایی با توجه به کتاب خانه نامپای ایجاد میکنیم
Datatotal = np.array(Datatotal)
matrixsafe = np.array(matrixsafe)
matrixspam = np.array(matrixspam)

# ////////////////////////////////////////////
#در این قسمت ارایه هایی ایجاد میکنیم برای ذخیره کردن احتمالات 3000 کلمه ی موجود در ایمیل ها حال این ایمیل می تواند اسپم یا از نوع غیر اسپم باشد که زمانی که مقدار کلمه غیر صفر باشد نشان دهنده استفاده شدن اون کلمه در ایمیل می باشد که در کل احتمال استفاده کلمه که در بخش اسپم شده ها و اسپم نشده ها بدست می آوریم
#در این بخش سعی شده حالت بدون اسپم و امن را بدست آورد که پس از  در داخل ارایه مربوطه قرار می دهد
probabmatrixspam = []
probabmatrixsafe = []

for i in range(1, 3001, 1):
    c = 0
    for j in range(1, 1500, 1):
        if (int(matrixsafe[j][i]) > 0):
            c = c + 1
    if (c == 0):
        cx = 0.0001
    else:
        cx = round(c / 1499, 4)
    probabmatrixsafe.append(cx)

probabmatrixsafe = np.array(probabmatrixsafe)
#///////////////////////////////////////////
#در این بخش برای بدست آوردن استفاده شدن هر کلمه در ایمیل هایی که اسپم صورت گرفته که احتمال حضور را با احتمال در ارایه ای قرار میدهد

for i in range(1, 3001, 1):
    c = 0
    for j in range(1, 3672, 1):
        if (int(matrixspam[j][i]) > 0):
            c = c + 1
    if (c == 0):

        cx = 0.0001
    else:
        cx = round(c / 3671, 4)
    probabmatrixspam.append(cx)

conttrue = 0
contfalse = 0
print(probabmatrixspam)
probabmatrixspam = np.array(probabmatrixspam)
print(probabmatrixspam.shape)


#///////////////////////////////
# در تابع زیر قرار است پیش بینی صورت گیرد 2000 ایمیل به عنوان تست  وارد می کنیم و به اضای وجود کلمه در داخل ایمیل احتمال ها را در هم ضرب میکنیم  و سر اخر احتمال اسپم ها و غیر اسپم ها و بعد برسی که احتمال کدام کوچک تر شده و کدام بزرگ تر است
#بر اساس بیشترین احتمال خروجی بدست می آید  که صفر یا یک با مقدار واقعی مقایسه می شود و با این کار میخواهیم بدست آوریم ایا پیش بینی درست بوده یا نادرست تا بتوان دقت شبکه را بیان کرد
outsafe = 1
outspam = 1
print(probabmatrixsafe.shape)
print(matrixsafe.shape)
#محاسبه احتمالات
for t in range(100, 2000, 1):
    outsafe = 1
    outspam = 1
    for i in range(0, 3000, 1):
        if (int(Datatotal[t][i + 1]) > 0):
            outsafe = outsafe * (probabmatrixsafe[i])
            outspam = outspam * (probabmatrixspam[i])


    outspam = outspam * (3672 / 5172)
    outsafe = outsafe * (1500 / 5172)
    #مقایسه احتمالات و دقت
    if (outspam > outsafe):
        op = 0
    else:
        op = 1
    print(op)
    if (op == int(Datatotal[t][3001])):
        conttrue += 1
    else:
        contfalse += 1
#خروجی دقت شبکه بر اساس کل داده های داده شده برای پیش بینی 
print(conttrue / 2000)
print(contfalse / 2000)
