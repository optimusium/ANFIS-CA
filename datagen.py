import os,re
import math,random

def eval1(a,b,c,d):
    e=0
    if 1:
        if a<0.37 and b>0.75 and c>0.34 and d<0.75 and d>0.25:
            e=1
        elif a>0.37 and a<0.7 and c>0.34 and d<0.75 and d>0.25:
            e=1
        elif a>0.37 and a<0.7 and b>0.75 and d<0.75 and d>0.25: # and (a<0.67 or c<0.67 or b<0.5):
            e=1
        elif a>0.7 and d<0.75 and d>0.25: # and (a<0.67 or c<0.67 or b<0.5):
            e=1
        else:
            e=0
    #e=0
    return e

def eval2(a,b,c,d):
    e=0
    if 1:
        if a<0.37 and b>0.75 and c>0.34 and d>0.75:
            e=1
        elif a>0.37 and a<0.7 and d>0.75:
            e=1
        elif a>0.7 and d>0.75: # and (a<0.67 or c<0.67 or b<0.5):
            e=1
        else:
            e=0
    #e=0
    return e
            
    

outf=open("Data.csv","w+")
outf.write("Seniority,Purchase_Propensity,CompanySize,Contactable,Action\n")
outf2=open("Data2.csv","w+")
outf2.write("Seniority,Purchase_Propensity,CompanySize,Contactable,Action\n")

outfa=open("Data_raw.csv","w+")
outfa.write("Seniority,Purchase_Propensity,CompanySize,Contactable,Action\n")
outfb=open("Data_raw2.csv","w+")
outfb.write("Seniority,Purchase_Propensity,CompanySize,Contactable,Action\n")




for i in range(3):
    
    #no one sending email
    for j in range(50):
        for k in range(20):
            aa=int(10*random.random())
            a=aa/9.0
            bb=int(3*random.random())
            b=bb/2.0
            cc=int(1501*random.random())
            c=cc/1500.0
            dd=int(3*random.random())
            d=dd/2.0
            e=eval1(a,b,c,d)
            outf.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )
            outfa.write("%s,%s,%s,%s,%s\n" % (aa,bb,cc,dd,e) )
            e=eval2(a,b,c,d)
            outf2.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )
            outfb.write("%s,%s,%s,%s,%s\n" % (aa,bb,cc,dd,e) )
            

    
outf.close()
outf2.close()
outfa.close()
outfb.close()

