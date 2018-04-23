all: host

device:
	nvcc -arch=sm_35 qr.cu -o qr_device.exe

host:
	gcc -Wall -Wshadow -g qr.c -o qr_host.exe -lm

clean:
	rm qr_device.exe
	rm qr_host.exe

