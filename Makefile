COMPUTE_CAPABILITY=35

all:
	nvcc -arch=sm_35 qr.cu -lcublas -o qr.exe

clean:
	rm qr.exe

