from flask import Flask, request, render_template  
import csv
  
 
app = Flask(__name__)    
  
 
@app.route('/', methods =["GET", "POST"]) 
def guest(): 
    if request.method == "POST": 
        
       name = request.form.get("name") 
       
       email = request.form.get("email")
       msg = request.form.get("message")

       fieldnames = ['name','email', 'msg']

       with open('nameList.csv','a',newline='') as inFile:
        writer = csv.DictWriter(inFile, fieldnames=fieldnames)

        writer.writerow({'name': name, 'email':email, 'msg': msg})

        inFile.close()



       return "Details Saved ! " 
    return render_template("form.html") 
  
if __name__=='__main__': 
   app.run() 
