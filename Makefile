all: device host

device:
	nvcc -arch=sm_35 qr.cu -o qr_device.exe

host:
	gcc -g qr.c -o qr_host.exe

clean:
	rm qr_device.exe
	rm qr_host.exe

