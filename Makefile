all: host

device:
	nvcc -arch=compute_35 -code=sm_35 -g -G -ccbin=icc qr.cu -o qr_device.exe

host: qr.c
	gcc -Wall -Wshadow -Wextra -g qr.c -o qr_host.exe -lm

clean:
	rm qr_device.exe
	rm qr_host.exe

