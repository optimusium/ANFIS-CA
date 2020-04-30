import os,re
import math,random

outf=open("Data.csv","w+")
outf.write("Age,Purchase_Propensity,CompanySize,Contactable,Action\n")

for i in range(3):
    #no one sending email
    for j in range(10):
        prob=random.random()
        a=0
        if prob<0.33:
            a=int(25+10*random.random())
        elif prob<0.66:
            a=int(35+7*random.random())
        else:
            a=int(42+18*random.random())
        prob=random.random()
        b=0
        if prob<0.33:
            b=3
        elif prob<0.66:
            b=12
        prob=random.random()
        c=0
        if prob<0.33:
            c=int(2+98*random.random())
        elif prob<0.66:
            c=int(101+399*random.random())
        else:
            c=int(501+1000*random.random())
        prob=random.random()
        d=0
        e=0
        #print(a,b,c,d,e)
        outf.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )

    #no email
    for j in range(10):
        prob=random.random()
        a=0
        if prob<0.8:
            a=int(25+12*random.random())
        elif prob<0.9:
            a=int(35+7*random.random())
        else:
            a=int(42+18*random.random())
        b=0
        prob=random.random()
        c=0
        if prob<0.8:
            c=int(2+98*random.random())
        elif prob<0.9:
            c=int(101+399*random.random())
        else:
            c=int(501+1000*random.random())
        d=1
        e=0
        #print(a,b,c,d,e)
        outf.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )

    #no email
    for j in range(10):
        prob=random.random()
        a=int(25+11*random.random())
        prob=random.random()
        if prob<0.15:
            b=3
        else:
            b=0
        prob=random.random()
        c=0
        if prob<0.75:
            c=int(2+98*random.random())
        elif prob<0.85:
            c=int(101+399*random.random())
        else:
            c=int(501+1000*random.random())
        d=2
        e=0
        #print(a,b,c,d,e)
        outf.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )

    #no email
    for j in range(10):
        prob=random.random()
        a=int(34+27*random.random())
        prob=random.random()
        b=0
        prob=random.random()
        c=0
        if prob<0.75:
            c=int(2+98*random.random())
        elif prob<0.85:
            c=int(101+399*random.random())
        else:
            c=int(501+1000*random.random())
        d=2
        e=0
        #print(a,b,c,d,e)
        outf.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )

    #brochure
    for j in range(10):
        prob=random.random()
        a=0
        if prob<0.15:
            a=int(25+10*random.random())
        elif prob<0.5:
            a=int(35+7*random.random())
        else:
            a=int(42+18*random.random())
        prob=random.random()
        b=0
        if prob<0.33:
            b=3
        else:
            b=1
        prob=random.random()
        c=0
        if prob<0.2:
            c=int(2+98*random.random())
        elif prob<0.66:
            c=int(101+399*random.random())
        else:
            c=int(501+1000*random.random())
        d=1
        e=1
        #print(a,b,c,d,e)
        outf.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )

    #brochure+email
    for j in range(10):
        prob=random.random()
        a=0
        if prob<0.4:
            a=int(33+10*random.random())
        else:
            a=int(43+18*random.random())
        prob=random.random()
        b=1
        prob=random.random()
        c=0
        if prob<0.15:
            c=int(2+98*random.random())
        elif prob<0.63:
            c=int(101+399*random.random())
        else:
            c=int(501+1000*random.random())
        d=2
        e=2
        #print(a,b,c,d,e)
        outf.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )

    #email
    for j in range(10):
        prob=random.random()
        a=0
        if prob<0.2:
            a=int(32+11*random.random())
        else:
            a=int(25+9*random.random())
        prob=random.random()
        b=3
        prob=random.random()
        c=0
        if prob<0.75:
            c=int(2+108*random.random())
        else:
            c=int(91+399*random.random())
        d=3
        e=3
        #print(a,b,c,d,e)
        outf.write("%s,%s,%s,%s,%s\n" % (a,b,c,d,e) )

outf.close()
