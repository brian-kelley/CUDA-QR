all: device host

device:
	nvcc -arch=sm_35 qr.cu -o qr_device.exe

host:
	gcc -std=c11 qr.c -o qr_host.exe

clean:
	rm qr_device.exe
	rm qr_host.exe

