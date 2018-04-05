COMPUTE_CAPABILITY=35

all:
	nvcc -arch=sm_35 qr.cu -o qr.exe

clean:
	rm qr.exe

