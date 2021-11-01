lab5: lab5.cu
	nvcc -arch=sm_30 -o lab5 lab5.cu
clean:
	rm lab5 
