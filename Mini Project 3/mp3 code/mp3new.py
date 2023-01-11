import time
import datetime
from tkinter import *
import tkinter.messagebox 

#Creating the main parent window using Tk() method-----
root=Tk()
root.title("Employee payroll manager")
root.geometry('1500x650')
root.configure(background="#1AEDED")    #LIGHT BLUE

#Creating first base frame on parent window-----
Mainframe=Frame(root,width=1350,height=50,bd=8,bg="#1AEDED")   
Mainframe.pack(side=TOP)

#Creating frame for Label and Entry widgets on parent window-----
f1=Frame(root,width=1200,height=600,bd=8,bg="#FF0000")  #RED
f1.pack(side=LEFT)

#Creating border frame for Payslip display widget on parent window-----
f2=Frame(root,width=300,height=700,bd=8,bg="#000000")
f2.pack(side=RIGHT)

#Creating inner frame on Frame f1-----
f3=Frame(f1,width=600,height=200,bd=8,bg="#000000")   #BLACK
f3.pack(side=TOP)

#Creating frame for Buttons on Frame f1-----
f4=Frame(f1,width=300,height=600,bd=8,bg="#8737ED")   #PURPLE
f4.pack(side=TOP)

#Creating System Heading Label at the top-----
#So one diff here is that for positioning of labels we use grid() method and not pack()-----
headlabel=Label(Mainframe,font=('arial',45,'bold'),text=" Payroll Management System ",bd=10,bg="#000000",fg="#10D7DE")
headlabel.grid(row=0,column=0)

# VARIABLES-----

Name=StringVar()
Address=StringVar()
HoursWorked=StringVar()
payperhour=StringVar()
Payable=StringVar()
TaxValue=StringVar()
NetPayable=StringVar()
GrossPayable=StringVar()
OverTimeBonus=StringVar()
Employer=StringVar()
EmployeeID=StringVar()
TimeOfOrder=StringVar()
Date=StringVar()

Date.set(time.strftime("%d/%m/%Y"))

# FUNCTION to Reset the system to take new values-----
def reset():
  Name.set("")
  Address.set("")
  HoursWorked.set("")
  payperhour.set("")
  Payable.set("")
  TaxValue.set("")
  NetPayable.set("")
  GrossPayable.set("")
  OverTimeBonus.set("")
  Employer.set("")
  EmployeeID.set("")
  txtpayslip.delete("1.0",END)

# FUNCTION to operate calculations in the system-----
def monthlypay():

  txtpayslip.delete("1.0",END)
  weeklyworkhours=float(HoursWorked.get())
  hourlywage=float(payperhour.get())
  
  # calculating overtime hours 
  # 40 is considered as 8 hours of work per day times 5 where, 5 is the number of working days in the week 
  overtimehours=(weeklyworkhours-40)
  
  #calculating overtime bonus
  #overtime pay is calculated using 1.5 times the regular pay 
  overtimepay=overtimehours*hourlywage*1.5
  
  #if overtime hours exist...
  if (overtimehours>0):
    otbonus="INR",str('%.2f'%(overtimepay))
    OverTimeBonus.set(otbonus)
    
    # we have subtracted (overtimehours*hourlywage) to avoid an error which was the adding the bonus 2 times
    grosspay=hourlywage*weeklyworkhours+overtimepay-(overtimehours*hourlywage) 
    paymentdue="INR",str('%.2f'%(grosspay))
    Payable.set(paymentdue)
  
  #if overtime hours doesnt exist...
  elif (overtimehours<=0):
    overtime="INR",str('%.2f'%(overtimepay))
    grosspay=hourlywage*weeklyworkhours
    paymentdue="INR",str('%.2f'%(grosspay))
    Payable.set(paymentdue)
  
  #calculating monthly tax here and multiplying by 4 where, 4 is no. of weeks in a months
  #In this case, 0.2 is used because tax is assumed to be 20% which is normally generalised by the govt 
  tax=grosspay*4*0.2
  taxed="INR",str('%.2f'%(tax))
  TaxValue.set(taxed)
  
  #calculating monthly net pay 
  netpay=(grosspay*4)-tax
  totalnetpay="INR",str('%.2f'%(netpay))
  NetPayable.set(totalnetpay)
     
# FUNCTION to Display Payslip in payslip widget-----
def payslipinfo():
  txtpayslip.delete("1.0",END)
  txtpayslip.insert(END,"\t\tPAY SLIP\n\n")
  txtpayslip.insert(END,"Name :\t\t"+Name.get()+"\n\n")
  txtpayslip.insert(END,"Address :\t\t"+Address.get()+"\n\n")
  txtpayslip.insert(END,"Employer :\t\t"+Employer.get()+"\n\n")
  txtpayslip.insert(END,"Employee ID :\t\t"+EmployeeID.get()+"\n\n")
  txtpayslip.insert(END,"Weekly Work Hours :\t\t"+HoursWorked.get()+"\n\n")
  txtpayslip.insert(END,"Hourly Pay :\t"+payperhour.get()+"\n\n")
  txtpayslip.insert(END,"Weekly Gross Pay :\t\t"+Payable.get()+"\n\n") 
  txtpayslip.insert(END,"Monthly Tax Paid :\t\t"+TaxValue.get()+"\n\n")
  txtpayslip.insert(END,"Monthly Net Pay :\t\t"+NetPayable.get()+"\n\n")

#FUNCTION to create exit button for the System-----
def exit():
  exit=tkinter.messagebox.askyesno("Employee system","Do you want to exit the system?")
  if exit>0:
    root.destroy()
    return

# LABEL WIDGETS-----

Namelabel=Label(f3,text="EMPLOYEE NAME",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=0,column=0)
Addresslabel=Label(f3,text="ADDRESS",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=0,column=2)
Employerlabel=Label(f3,text="EMPLOYER",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=1,column=0)
EmployeeIDlabel=Label(f3,text="EMPLOYEE ID",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=1,column=2)
HoursWorkedlabel=Label(f3,text="WEEKLY WORK HRS",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=2,column=0)
HourlyPaylabel=Label(f3,text="HOURLY PAY",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=2,column=2)
Taxlabel=Label(f3,text="MONTHLY TAX RATE",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=3,column=0)
OverTimelabel=Label(f3,text="OVERTIME PAY",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=3,column=2)
GrossPaylabel=Label(f3,text="WEEKLY GROSS PAY",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=4,column=0)
NetPaylabel=Label(f3,text="MONTHLY NET PAY",font=('arial',16,'bold'),bd=20,fg="#FF0000",bg="#000000").grid(row=4,column=2)


# ENTRY WIDGETS-----

empname=Entry(f3,textvariable=Name,font=('arial',16,'bold'),bd=16,width=22,justify='left')
empname.grid(row=0,column=1)

empaddress=Entry(f3,textvariable=Address,font=('arial',16,'bold'),bd=16,width=22,justify='left')
empaddress.grid(row=0,column=3)

empemployer=Entry(f3,textvariable=Employer,font=('arial',16,'bold'),bd=16,width=22,justify='left')
empemployer.grid(row=1,column=1)

empworkhours=Entry(f3,textvariable=HoursWorked,font=('arial',16,'bold'),bd=16,width=22,justify='left')
empworkhours.grid(row=2,column=1)

emppayperhour=Entry(f3,textvariable=payperhour,font=('arial',16,'bold'),bd=16,width=22,justify='left')
emppayperhour.grid(row=2,column=3)

empid=Entry(f3,textvariable=EmployeeID,font=('arial',16,'bold'),bd=16,width=22,justify='left')
empid.grid(row=1,column=3)

empgrosspay=Entry(f3,textvariable=Payable,font=('arial',16,'bold'),bd=16,width=22,justify='left')
empgrosspay.grid(row=4,column=1)

empnetpay=Entry(f3,textvariable=NetPayable,font=('arial',16,'bold'),bd=16,width=22,justify='left')
empnetpay.grid(row=4,column=3)

emptax=Entry(f3,textvariable=TaxValue,font=('arial',16,'bold'),bd=16,width=22,justify='left')
emptax.grid(row=3,column=1)

empovertime=Entry(f3,textvariable=OverTimeBonus,font=('arial',16,'bold'),bd=16,width=22,justify='left')
empovertime.grid(row=3,column=3)

# CREATING PAYSLIP WIDGET-----

Datelabel=Label(f2,textvariable=Date,font=('arial',21,'bold'),fg="#FFFF00",bg="#000000").grid(row=0,column=0)
txtpayslip=Text(f2,height=22,width=34,bd=16,font=('arial',13,'bold'),fg="#0000FF",bg="#FFFFFF")
txtpayslip.grid(row=1,column=0)

# CREATING BUTTONS ON APPLICATION-----

salarybutton=Button(f4,text='Monthly Salary',padx=10,pady=10,bd=8,font=('arial',16,'bold'),width=14,fg="#FF0000",bg="#1AEDED",command=monthlypay).grid(row=0,column=0)

resetbutton=Button(f4,text='Reset System',padx=10,pady=10,bd=8,font=('arial',16,'bold'),width=14,command=reset,fg="#FF0000",bg="#1AEDED").grid(row=0,column=1)

payslipbutton=Button(f4,text='Print Payslip',padx=10,pady=10,bd=8,font=('arial',16,'bold'),width=14,command=payslipinfo,fg="#FF0000",bg="#1AEDED").grid(row=0,column=2)

exitbutton=Button(f4,text='Exit System',padx=10,pady=10,bd=8,font=('arial',16,'bold'),width=14,command=exit,fg="#FF0000",bg="#1AEDED").grid(row=0,column=3)

root.mainloop()