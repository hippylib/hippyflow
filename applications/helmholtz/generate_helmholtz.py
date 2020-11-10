import os
import time
import numpy as np

def send_job_update(message):
	assert type(message) == str
	import smtplib
	mail = smtplib.SMTP('smtp.gmail.com',587)
	mail.ehlo()
	mail.starttls()
	mail.login('thejobfinished@gmail.com',os.environ['JOBSDONEPWORD'])
	mail.sendmail('thejobfinished@gmail.com','tom.olearyroseberry@gmail.com',message) 
	mail.close()



# gds = [(1.0,10.0),(0.5,2.5),(0.5,5.0),(1.,5.),(2.,10.)]

gds = [(1.,5.)]


# gds = [(0.1,0.5),(0.5,2.5),(1.0,1.0),(5.0,1.0),(0.5,5.0)]

# for gamma in gammas:
# 	for delta in deltas:
# 		gds.append((gamma,delta))

# nxnys = [(32,32),(64,64),(96,96),(128,128),(160,160),(192,192)]
nxnys = [(128,128)]
# nxnys = [(128,128)]
frequencies = [600,800]

for (gamma,delta) in gds:
	t0 = time.time()
	for nx,ny in nxnys:
		for frequency in frequencies:
			print(80*'#')
			print(('Running for gd = '+str((gamma,delta))+' nx,ny = '+str((nx,ny))+' f '+str(frequency)).center(80))
			os.system('mpirun -n 4 python helmholtz_problem_setup.py -ninstance 4 -gamma '\
					+str(gamma)+' -delta '+str(delta)+' -nx '+str(nx)+' -ny '+str(ny)+' -frequency '+str(frequency))
			email_text = 'Finished the data generation job for gamma,delta = '+str((gamma,delta))+' and it took '+str(time.time() - t0)+' s \n'
			email_text += 'Frequency is '+str(frequency)
			email_subject = 'Helmholtz data generation'+' gamma,delta = '+str((gamma,delta))
			message = 'Subject: {}\n\n{}'.format(email_subject, email_text)
			send_job_update(message)
